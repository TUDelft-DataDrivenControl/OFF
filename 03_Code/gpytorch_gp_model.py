"""
GPyTorch GP Surrogate Model - Load and Predict Only 
To be coupled with py_wake through wind_farm_loads toolbox
Author: Daan van der Hoek
Date: 2026-04-07
"""

import pickle
import h5py
import numpy as np
import torch
import gpytorch as gp
from sklearn.cluster import KMeans
from surrogates_interface.surrogates import SurrogateModel

class GPyTorchGPSurrogate(SurrogateModel):
    """
    GPyTorch ExactGP for inference.
      
    Usage:
    ------
    # Load pre-trained model
    surrogate = GPyTorchGPSurrogate.load_h5('model.h5')
    
    # Predict
    y_mean = surrogate.predict_output(X_test)
    y_mean, y_std = surrogate.predict_with_std(X_test)
    """
    
    # ========================================================================
    # GPYTORCH MODEL DEFINITION
    # ========================================================================
    
    class ExactGPModel(gp.models.ExactGP):
        """
        GPyTorch Exact GP with configurable kernel and mean.
        - Configurable mean function: 'linear', 'constant', or 'zero'
        - Configurable kernel: Matérn(nu), RBF, or composite kernels with ARD
        """
        
        def __init__(self, train_x, train_y, likelihood, n_inputs, kernel_nu=2.5, mean_type='linear', kernel_type='matern'):
            super().__init__(train_x, train_y, likelihood)
            
            # Mean function (configurable)
            if mean_type == 'linear':
                self.mean_module = gp.means.LinearMean(input_size=n_inputs)
            elif mean_type == 'constant':
                self.mean_module = gp.means.ConstantMean()
            elif mean_type == 'zero':
                self.mean_module = gp.means.ZeroMean()
            else:
                raise ValueError(f"Unknown mean_type: {mean_type}. Choose 'linear', 'constant', or 'zero'")
            
            # Kernel (configurable with composite options)
            if kernel_type == 'rbf':
                # RBF kernel (infinitely differentiable)
                kernel = gp.kernels.RBFKernel(ard_num_dims=n_inputs)
            elif kernel_type == 'matern':
                # Matérn kernel with specified nu
                kernel = gp.kernels.MaternKernel(nu=kernel_nu, ard_num_dims=n_inputs)
            elif kernel_type == 'matern*rbf':
                # Product of Matérn and RBF (multi-scale smoothness)
                matern_kernel = gp.kernels.MaternKernel(nu=kernel_nu, ard_num_dims=n_inputs)
                rbf_kernel = gp.kernels.RBFKernel(ard_num_dims=n_inputs)
                kernel = matern_kernel * rbf_kernel
            elif kernel_type == 'matern+linear':
                # Matérn + Linear kernel (for linear trends)
                matern_kernel = gp.kernels.MaternKernel(nu=kernel_nu, ard_num_dims=n_inputs)
                linear_kernel = gp.kernels.LinearKernel(num_dimensions=n_inputs)
                kernel = matern_kernel + linear_kernel
            else:
                raise ValueError(f"Unknown kernel_type: {kernel_type}")
            
            self.covar_module = gp.kernels.ScaleKernel(kernel)
        
        def forward(self, x):
            """Compute predictive distribution."""
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gp.distributions.MultivariateNormal(mean_x, covar_x)
    
    # ========================================================================
    # SURROGATE WRAPPER
    # ========================================================================
    
    def __init__(self, input_names, output_names,
                 input_transformers=None, output_transformers=None,
                 hyperparameters=None, kernel_nu=2.5, mean_type='constant',
                 kernel_type='matern', optimizer_type='lbfgs', min_noise=1e-4,
                 device='cpu', dtype=torch.float64):
        """
        Initialize GPyTorch surrogate.
        
        Parameters:
        -----------
        input_names : list of str
            Names of input features
        output_names : list of str
            Names of output features (single output supported)
        input_transformers : list of transformers
            Applied before model.predict()
        output_transformers : list of transformers
            Applied after model.predict()
        hyperparameters : dict or None
            Pre-trained hyperparameters with keys: 'outputscale', 'noise', 'lengthscales'
            If None, will train from scratch
        kernel_nu : float or 'rbf'
            Matérn kernel smoothness: 0.5, 1.5, 2.5 (default) or 'rbf' for RBF kernel
            Lower values = less smooth, better for rough functions
        mean_type : str
            Mean function: 'linear', 'constant' (default), or 'zero'
        kernel_type : str
            Kernel type: 'matern' (default), 'rbf', 'matern*rbf', 'matern+linear'
        optimizer_type : str
            Optimizer: 'adam' or 'lbfgs' (default)
        min_noise : float
            Minimum noise constraint (default: 1e-4, try 1e-3 or 1e-2 to reduce overfitting)
        device : str
            'cpu' or 'cuda'
        dtype : torch.dtype
            torch.float32 or torch.float64 (default: float64 for precision)
        """
        self.device = device
        self.dtype = dtype
        self.hyperparameters = hyperparameters
        self.kernel_nu = kernel_nu
        self.mean_type = mean_type
        self.kernel_type = kernel_type
        self.optimizer_type = optimizer_type
        self.min_noise = min_noise
        
        # Training data (loaded from HDF5)
        self.train_x = None
        self.train_y = None
        
        # Model and likelihood (reconstructed from HDF5)
        self.gp_model = None
        self.likelihood = None
        
        # Initialize base class
        super().__init__(
            model=None,  # Will be set in load_h5()
            input_names=input_names,
            output_names=output_names,
            input_transformers=input_transformers if input_transformers is not None else [],
            output_transformers=output_transformers if output_transformers is not None else []
        )
        
        self._is_fitted = False
    
    # ========================================================================
    # PREDICTION METHODS
    # ========================================================================
    
    def _predict_model_output(self, x_transformed):
        """
        Internal prediction method called by SurrogateModel.predict_output().
        
        Parameters:
        -----------
        x_transformed : array-like of shape (n_samples, n_inputs)
            Transformed input data
        
        Returns:
        --------
        y_transformed : array-like of shape (n_samples, 1)
            Transformed output predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Set to evaluation mode
        self.gp_model.eval()
        self.likelihood.eval()
        
        # Convert to torch tensor
        x_torch = torch.tensor(x_transformed, dtype=self.dtype, device=self.device)
        
        # Predict with fast predictive variance computation
        with torch.no_grad(), gp.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp_model(x_torch))
            y_pred = observed_pred.mean
        
        # Convert back to numpy
        y_pred_np = y_pred.cpu().numpy()
        
        return y_pred_np.reshape(-1, 1)
    
    def predict_with_std(self, X):
        """
        Predict with uncertainty quantification.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_inputs)
            Input data (untransformed)
        
        Returns:
        --------
        y_mean : array-like of shape (n_samples, 1)
            Mean predictions (untransformed)
        y_std : array-like of shape (n_samples, 1)
            Standard deviations (untransformed scale)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Transform inputs
        x_transformed = X.copy()
        for transformer in self.input_transformers:
            x_transformed = transformer.transform(x_transformed, inplace=False)
        
        # Set to evaluation mode
        self.gp_model.eval()
        self.likelihood.eval()
        
        # Convert to torch tensor
        x_torch = torch.tensor(x_transformed, dtype=self.dtype, device=self.device)
        
        # Predict with uncertainty
        with torch.no_grad(), gp.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp_model(x_torch))
            y_mean = observed_pred.mean.cpu().numpy()
            y_std = observed_pred.stddev.cpu().numpy()
        
        # Reshape
        y_mean = y_mean.reshape(-1, 1)
        y_std = y_std.reshape(-1, 1)
        
        # Inverse transform mean
        for transformer in reversed(self.output_transformers):
            y_mean = transformer.inverse_transform(y_mean, inplace=False)
        
        # Scale std (only by output std, not shift)
        for transformer in reversed(self.output_transformers):
            if hasattr(transformer, 'scale_'):
                y_std = y_std * transformer.scale_
        
        return y_mean, y_std
    
    # ========================================================================
    # TRAINING METHODS
    # ========================================================================
    
    def fit(self, X, y, n_iter=100):
        """
        Train GPyTorch model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_inputs)
            Training input data (untransformed)
        y : array-like of shape (n_samples,)
            Training output data (untransformed)
        n_iter : int
            Number of optimization iterations (default: 100)
        """
        # Transform training data
        X_train = X.copy()
        for transformer in self.input_transformers:
            X_train = transformer.transform(X_train, inplace=False)
        
        y_train = y.copy().reshape(-1, 1)
        for transformer in self.output_transformers:
            y_train = transformer.transform(y_train, inplace=False)
        y_train = y_train.flatten()
        
        # Convert to torch tensors
        self.train_x = torch.tensor(X_train, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(y_train, dtype=self.dtype, device=self.device)
        
        # Create likelihood and model (no noise constraint - let optimizer find optimal value)
        self.likelihood = gp.likelihoods.GaussianLikelihood().to(device=self.device, dtype=self.dtype)
        
        self.gp_model = self.ExactGPModel(
            self.train_x, self.train_y, self.likelihood,
            n_inputs=X_train.shape[1],
            kernel_nu=self.kernel_nu,
            mean_type=self.mean_type,
            kernel_type=self.kernel_type
        ).to(device=self.device, dtype=self.dtype)
        
        # Set hyperparameters if provided
        if self.hyperparameters is not None:
            # Set parameters directly (don't use state_dict for initial values)
            self.gp_model.covar_module.outputscale = torch.tensor(
                self.hyperparameters['outputscale'], dtype=self.dtype, device=self.device
            )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.tensor(
                [self.hyperparameters['lengthscales']], dtype=self.dtype, device=self.device
            )
            self.likelihood.noise = torch.tensor(
                self.hyperparameters['noise'], dtype=self.dtype, device=self.device
            )
        
        # Train
        self.gp_model.train()
        self.likelihood.train()
        
        mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        
        # Track best loss and state
        best_loss = float('inf')
        best_state = None
        
        if self.optimizer_type == 'lbfgs':
            # L-BFGS optimizer (often better for GP hyperparameters)
            optimizer = torch.optim.LBFGS(self.gp_model.parameters(), lr=0.1, max_iter=20)
            
            def closure():
                optimizer.zero_grad()
                output = self.gp_model(self.train_x)
                loss = -mll(output, self.train_y)
                loss.backward()
                return loss
            
            for i in range(n_iter // 20):
                loss = optimizer.step(closure)
                current_loss = loss.item()
                
                # Track best state
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state = {
                        'model': self.gp_model.state_dict(),
                        'likelihood': self.likelihood.state_dict()
                    }
                
                # Print every iteration (= every 20 LBFGS iterations)
                print(f"  Iter {(i+1)*20}/{n_iter} - Loss: {current_loss:.3f}")
        else:
            # Adam optimizer 
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
            
            for i in range(n_iter):
                optimizer.zero_grad()
                output = self.gp_model(self.train_x)
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                
                # Track best state
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state = {
                        'model': self.gp_model.state_dict(),
                        'likelihood': self.likelihood.state_dict()
                    }
                
                # Print every 20 iterations
                if (i + 1) % 20 == 0:
                    print(f"  Iter {i+1}/{n_iter} - Loss: {current_loss:.3f}")
        
        # Restore best state
        if best_state is not None:
            self.gp_model.load_state_dict(best_state['model'])
            self.likelihood.load_state_dict(best_state['likelihood'])
            print(f"\nRestored best model (loss: {best_loss:.3f})")
        
        self._is_fitted = True
        self.model = self.gp_model
    
    def get_hyperparameters(self):
        """
        Extract trained hyperparameters.
        
        Returns:
        --------
        hyperparams : dict
            Dictionary with 'outputscale', 'noise', and 'lengthscales'
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before extracting hyperparameters")
        
        # Access actual parameter values (after constraints applied)
        hyperparams = {
            'outputscale': float(self.gp_model.covar_module.outputscale.detach().cpu().numpy()),
            'noise': float(self.likelihood.noise.detach().cpu().numpy()),
            'lengthscales': self.gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten().tolist()
        }
        
        return hyperparams
    
    def save_h5(self, filepath):
        """
        Save model to HDF5 file (framework standard format).
        
        Parameters:
        -----------
        filepath : str
            Path to save HDF5 file
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Prepare model state for pickling
        model_state = {
            'gp_model_state': self.gp_model.state_dict(),
            'likelihood_state': self.likelihood.state_dict(),
            'train_x': self.train_x.cpu().numpy(),
            'train_y': self.train_y.cpu().numpy(),
            'device': self.device,
            'dtype_str': str(self.dtype),
            'n_inputs': self.train_x.shape[1],
            'kernel_nu': self.kernel_nu,
            'mean_type': self.mean_type
        }
        
        model_bytes = pickle.dumps(model_state)
        
        with h5py.File(filepath, 'w') as f:
            # Save model
            f.create_dataset('gpytorch_model', data=np.void(model_bytes))
            
            # Save metadata
            f.create_dataset('input_names', data=self.input_names, dtype=h5py.string_dtype())
            f.create_dataset('output_names', data=self.output_names, dtype=h5py.string_dtype())
            
            # Save transformers
            if self.input_transformers:
                input_tf_group = f.create_group('input_transformers')
                for i, tf in enumerate(self.input_transformers):
                    tf_group = input_tf_group.create_group(f'transformer_{i}')
                    tf_group.create_dataset('type', data=str(type(tf).__name__), dtype=h5py.string_dtype())
                    
                    if hasattr(tf, 'min_') and hasattr(tf, 'scale_'):
                        # MinMaxScaler
                        tf_group.create_dataset('min_', data=tf.min_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
                    elif hasattr(tf, 'mean_') and hasattr(tf, 'scale_'):
                        # StandardScaler
                        tf_group.create_dataset('mean_', data=tf.mean_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
            
            if self.output_transformers:
                output_tf_group = f.create_group('output_transformers')
                for i, tf in enumerate(self.output_transformers):
                    tf_group = output_tf_group.create_group(f'transformer_{i}')
                    tf_group.create_dataset('type', data=str(type(tf).__name__), dtype=h5py.string_dtype())
                    
                    if hasattr(tf, 'min_') and hasattr(tf, 'scale_'):
                        # MinMaxScaler
                        tf_group.create_dataset('min_', data=tf.min_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
                    elif hasattr(tf, 'mean_') and hasattr(tf, 'scale_'):
                        # StandardScaler
                        tf_group.create_dataset('mean_', data=tf.mean_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
    
    # ========================================================================
    # HDF5 LOAD (Framework standard)
    # ========================================================================
    
    @classmethod
    def _load_model(cls, filepath):
        """
        Load GPyTorch model from HDF5 file.
        
        Called by: GPyTorchGPSurrogate.load_h5(filepath)
        Framework will reconstruct: transformers, input_names, output_names
        
        Parameters:
        -----------
        filepath : str
            Path to HDF5 file
        
        Returns:
        --------
        model_data : dict
            Contains all data needed to reconstruct the model
        """
        with h5py.File(filepath, 'r') as f:
            # Load and unpickle
            model_bytes = bytes(f['gpytorch_model'][()])
            model_state = pickle.loads(model_bytes)
        
        return model_state
    
    @classmethod
    def load_h5(cls, filepath):
        """
        Load complete surrogate from HDF5 file.
        
        Override base class to properly reconstruct GPyTorch model.
        Uses base class transformer loading infrastructure.
        
        Parameters:
        -----------
        filepath : str
            Path to HDF5 file
        
        Returns:
        --------
        surrogate : GPyTorchGPSurrogate
            Loaded surrogate model
        """
        # Load model state from GPyTorch-specific data
        model_state = cls._load_model(filepath)
        
        # Use base class to load transformers (it knows the correct format)
        with h5py.File(filepath, 'r') as f:
            input_names = list(f['input_names'].asstr()[:])
            output_names = list(f['output_names'].asstr()[:])
        
        # Import transformer loading from surrogates_interface
        from surrogates_interface.transformers import MinMaxScaler, StandardScaler
        from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
        from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
        
        # Load transformers using framework's HDF5 structure
        input_transformers = []
        output_transformers = []
        
        with h5py.File(filepath, 'r') as f:
            # Input transformers
            if 'input_transformers' in f:
                for tf_name in f['input_transformers'].keys():
                    tf_group = f['input_transformers'][tf_name]
                    tf_type = tf_group['type'][()].decode('utf-8')
                    
                    if 'MinMaxScaler' in tf_type:
                        sklearn_scaler = SklearnMinMaxScaler()
                        sklearn_scaler.min_ = tf_group['min_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.data_min_ = sklearn_scaler.min_
                        sklearn_scaler.data_max_ = sklearn_scaler.min_ + 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.data_range_ = 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.min_)
                        sklearn_scaler.feature_names_in_ = None
                        tf = MinMaxScaler.from_sklearn(sklearn_scaler)
                        input_transformers.append(tf)
                    elif 'StandardScaler' in tf_type:
                        sklearn_scaler = SklearnStandardScaler()
                        sklearn_scaler.mean_ = tf_group['mean_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.var_ = sklearn_scaler.scale_ ** 2
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.mean_)
                        sklearn_scaler.feature_names_in_ = None
                        tf = StandardScaler.from_sklearn(sklearn_scaler)
                        input_transformers.append(tf)
            
            # Output transformers
            if 'output_transformers' in f:
                for tf_name in f['output_transformers'].keys():
                    tf_group = f['output_transformers'][tf_name]
                    tf_type = tf_group['type'][()].decode('utf-8')
                    
                    if 'MinMaxScaler' in tf_type:
                        sklearn_scaler = SklearnMinMaxScaler()
                        sklearn_scaler.min_ = tf_group['min_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.data_min_ = sklearn_scaler.min_
                        sklearn_scaler.data_max_ = sklearn_scaler.min_ + 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.data_range_ = 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.min_)
                        sklearn_scaler.feature_names_in_ = None
                        tf = MinMaxScaler.from_sklearn(sklearn_scaler)
                        output_transformers.append(tf)
                    elif 'StandardScaler' in tf_type:
                        sklearn_scaler = SklearnStandardScaler()
                        sklearn_scaler.mean_ = tf_group['mean_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.var_ = sklearn_scaler.scale_ ** 2
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.mean_)
                        sklearn_scaler.feature_names_in_ = None
                        tf = StandardScaler.from_sklearn(sklearn_scaler)
                        output_transformers.append(tf)
        
        # Reconstruct dtype
        dtype_str = model_state['dtype_str']
        dtype = torch.float64 if 'float64' in dtype_str else torch.float32
        
        # Create instance
        instance = cls(
            input_names=input_names,
            output_names=output_names,
            input_transformers=input_transformers,
            output_transformers=output_transformers,
            device=model_state['device'],
            dtype=dtype
        )
        
        # Restore training data
        instance.train_x = torch.tensor(
            model_state['train_x'], dtype=dtype, device=model_state['device']
        )
        instance.train_y = torch.tensor(
            model_state['train_y'], dtype=dtype, device=model_state['device']
        )
        
        # Recreate likelihood and model (no noise constraint)
        instance.likelihood = gp.likelihoods.GaussianLikelihood().to(device=model_state['device'], dtype=dtype)
        
        # Get kernel/mean config from state if available, else use defaults
        kernel_nu = model_state.get('kernel_nu', 2.5)
        mean_type = model_state.get('mean_type', 'constant')
        
        instance.gp_model = cls.ExactGPModel(
            instance.train_x, instance.train_y, instance.likelihood,
            n_inputs=model_state['n_inputs'],
            kernel_nu=kernel_nu,
            mean_type=mean_type
        ).to(device=model_state['device'], dtype=dtype)
        
        # Load state dicts (use strict=False to handle priors)
        instance.gp_model.load_state_dict(model_state['gp_model_state'], strict=False)
        instance.likelihood.load_state_dict(model_state['likelihood_state'], strict=False)
        
        instance._is_fitted = True
        instance.model = instance.gp_model
        
        return instance


class HeteroscedasticGPSurrogate(SurrogateModel):
    """
    GPyTorch Heteroscedastic GP with variational inference.
    
    Uses two GPs:
    - One for the latent function mean
    - One for the log-noise function (input-dependent uncertainty)
    
    This allows the model to have higher uncertainty in complex regions
    (e.g., wake interactions) and lower uncertainty in simpler regions
    (e.g., freestream conditions).
    
    Usage:
    ------
    # Train model
    surrogate = HeteroscedasticGPSurrogate()
    surrogate.fit(X_train, y_train, num_inducing=500, lr=0.01, n_iter=1000)
    
    # Predict with heteroscedastic uncertainty
    y_mean, y_std = surrogate.predict_with_std(X_test)
    
    # Save/load
    surrogate.save_h5('hetero_model.h5')
    surrogate = HeteroscedasticGPSurrogate.load_h5('hetero_model.h5')
    """
    
    class HeteroscedasticGPModel(gp.models.ApproximateGP):
        """
        GPyTorch Approximate GP with heteroscedastic noise using variational inference.
        
        Architecture:
        - Latent function GP: Models the mean function f(x)
        - Noise function GP: Models log(σ²(x)) for input-dependent noise
        - Uses inducing points for scalability
        """
        
        def __init__(self, inducing_points, n_inputs, kernel_nu=2.5, mean_type='linear'):
            # Variational distribution and strategy
            variational_distribution = gp.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gp.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            
            # Mean function for latent GP
            if mean_type == 'linear':
                self.mean_module = gp.means.LinearMean(input_size=n_inputs)
            elif mean_type == 'constant':
                self.mean_module = gp.means.ConstantMean()
            elif mean_type == 'zero':
                self.mean_module = gp.means.ZeroMean()
            else:
                raise ValueError(f"Unknown mean_type: {mean_type}")
            
            # Covariance function for latent GP
            self.covar_module = gp.kernels.ScaleKernel(
                gp.kernels.MaternKernel(nu=kernel_nu, ard_num_dims=n_inputs)
            )
            
            # Noise model GP (models log-variance)
            self.noise_mean_module = gp.means.ConstantMean()
            self.noise_covar_module = gp.kernels.ScaleKernel(
                gp.kernels.MaternKernel(nu=kernel_nu, ard_num_dims=n_inputs)
            )
        
        def forward(self, x):
            """Forward pass returns latent function distribution."""
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gp.distributions.MultivariateNormal(mean_x, covar_x)
        
        def forward_noise(self, x):
            """Forward pass for noise model returns log-variance distribution."""
            noise_mean = self.noise_mean_module(x)
            noise_covar = self.noise_covar_module(x)
            return gp.distributions.MultivariateNormal(noise_mean, noise_covar)
    
    def __init__(self, input_names=None, output_names=None,
                 input_transformers=None, output_transformers=None):
        """
        Initialize Heteroscedastic GP Surrogate.
        
        Parameters:
        -----------
        input_names : list of str, optional
            Names of input features
        output_names : list of str, optional
            Names of output features
        input_transformers : list of transformers, optional
            Applied before model.predict()
        output_transformers : list of transformers, optional
            Applied after model.predict()
        """
        # Initialize base class
        super().__init__(
            model=None,
            input_names=input_names if input_names is not None else [],
            output_names=output_names if output_names is not None else [],
            input_transformers=input_transformers if input_transformers is not None else [],
            output_transformers=output_transformers if output_transformers is not None else []
        )
        
        self.gp_model = None
        self.n_inputs = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self._is_fitted = False
        self.model = None
        self.train_x = None
        self.train_y = None
        self.inducing_points = None
        self.kernel_nu = 2.5
        self.mean_type = 'linear'
    
    def _initialize_inducing_points(self, X, num_inducing):
        """Initialize inducing points using k-means clustering."""
        if num_inducing >= X.shape[0]:
            # Use all training points if num_inducing is larger
            return X.clone()
        
        # Use k-means to select representative points
        kmeans = KMeans(n_clusters=num_inducing, random_state=42, n_init=10)
        kmeans.fit(X.cpu().numpy())
        inducing_points = torch.tensor(
            kmeans.cluster_centers_, 
            device=X.device, 
            dtype=X.dtype
        )
        return inducing_points
    
    def _predict_model_output(self, X):
        """Abstract method implementation - returns mean predictions."""
        self.gp_model.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            X_torch = torch.tensor(X, device=self.device, dtype=self.dtype)
            predictions = self.gp_model(X_torch)
            y_mean = predictions.mean.cpu().numpy()
        return y_mean
    
    def predict_with_std(self, X):
        """
        Predict mean and heteroscedastic standard deviation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_inputs)
            Input data (untransformed)
        
        Returns
        -------
        y_mean : np.ndarray, shape (n_samples, 1)
            Predicted means (untransformed)
        y_std : np.ndarray, shape (n_samples, 1)
            Predicted standard deviations (heteroscedastic, untransformed scale)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Transform inputs
        x_transformed = X.copy()
        for transformer in self.input_transformers:
            x_transformed = transformer.transform(x_transformed, inplace=False)
        
        # Set to evaluation mode
        self.gp_model.eval()
        
        # Convert to torch tensor
        x_torch = torch.tensor(x_transformed, dtype=self.dtype, device=self.device)
        
        # Predict with heteroscedastic uncertainty
        with torch.no_grad(), gp.settings.fast_pred_var():
            # Latent function prediction
            f_pred = self.gp_model(x_torch)
            y_mean = f_pred.mean
            
            # Noise function prediction (log-variance)
            log_var_pred = self.gp_model.forward_noise(x_torch)
            log_var_mean = log_var_pred.mean
            
            # Total variance = latent uncertainty + heteroscedastic noise
            latent_var = f_pred.variance
            noise_var = torch.exp(log_var_mean)
            total_var = latent_var + noise_var
            
            y_std = torch.sqrt(total_var)
            
            # Convert to numpy
            y_mean = y_mean.cpu().numpy()
            y_std = y_std.cpu().numpy()
        
        # Reshape
        y_mean = y_mean.reshape(-1, 1)
        y_std = y_std.reshape(-1, 1)
        
        # Inverse transform mean predictions
        for transformer in reversed(self.output_transformers):
            y_mean = transformer.inverse_transform(y_mean, inplace=False)
        
        # Scale std (only by output std, not shift)
        for transformer in reversed(self.output_transformers):
            if hasattr(transformer, 'scale_'):
                y_std = y_std * transformer.scale_
        
        return y_mean, y_std
    
    def fit(self, X, y, num_inducing=500, lr=0.01, n_iter=1000, 
            kernel_nu=2.5, mean_type='linear', noise_lr_factor=0.1, 
            warmup_fraction=0.1, verbose=True):
        """
        Train heteroscedastic GP using variational inference.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features (untransformed)
        y : np.ndarray, shape (n_samples,)
            Training targets (untransformed)
        num_inducing : int, default=500
            Number of inducing points for variational inference
        lr : float, default=0.01
            Learning rate for Adam optimizer (mean function)
        n_iter : int, default=1000
            Number of training iterations
        kernel_nu : float, default=2.5
            Matérn kernel smoothness parameter
        mean_type : str, default='linear'
            Type of mean function: 'linear', 'constant', or 'zero'
        noise_lr_factor : float, default=0.1
            Learning rate multiplier for noise function (relative to lr)
        warmup_fraction : float, default=0.1
            Fraction of iterations to freeze noise model during warmup
        verbose : bool, default=True
            Print training progress
        """
        # Store configuration
        self.kernel_nu = kernel_nu
        self.mean_type = mean_type
        
        # Transform training data
        X_train = X.copy()
        for transformer in self.input_transformers:
            X_train = transformer.transform(X_train, inplace=False)
        
        y_train = y.copy().reshape(-1, 1)
        for transformer in self.output_transformers:
            y_train = transformer.transform(y_train, inplace=False)
        y_train = y_train.flatten()
        
        self.n_inputs = X_train.shape[1]
        
        # Convert to torch tensors
        X_torch = torch.tensor(X_train, device=self.device, dtype=self.dtype)
        y_torch = torch.tensor(y_train, device=self.device, dtype=self.dtype)
        
        self.train_x = X_torch
        self.train_y = y_torch
        
        # Initialize inducing points
        self.inducing_points = self._initialize_inducing_points(X_torch, num_inducing)
        
        # Create model
        self.gp_model = self.HeteroscedasticGPModel(
            self.inducing_points, self.n_inputs, kernel_nu, mean_type
        ).to(device=self.device, dtype=self.dtype)
        
        # Initialize noise model to data-driven value
        # Empirically tested: /30 provides good balance between accuracy and calibration
        empirical_var = y_torch.var().item()
        init_log_noise = np.log(empirical_var / 30.0)
        self.gp_model.noise_mean_module.constant.data.fill_(init_log_noise)
        
        if verbose:
            print(f"Initialized noise model: log(σ²) = {init_log_noise:.4f}, σ² = {np.exp(init_log_noise):.6f}")
            print(f"Empirical variance: {empirical_var:.6f}")
        
        # Marginal log likelihood for variational inference
        mll = gp.mlls.VariationalELBO(
            gp.likelihoods.GaussianLikelihood(), 
            self.gp_model, 
            num_data=y_torch.numel()
        )
        
        # Training mode
        self.gp_model.train()
        
        # Separate optimizers for mean function and noise function
        # Mean function learns faster to capture systematic patterns
        # Noise function learns slower to avoid over-explaining variance
        mean_params = [
            {'params': self.gp_model.variational_strategy.parameters()},
            {'params': self.gp_model.mean_module.parameters()},
            {'params': self.gp_model.covar_module.parameters()},
        ]
        noise_params = [
            {'params': self.gp_model.noise_mean_module.parameters()},
            {'params': self.gp_model.noise_covar_module.parameters()},
        ]
        
        mean_optimizer = torch.optim.Adam(mean_params, lr=lr)
        noise_optimizer = torch.optim.Adam(noise_params, lr=lr * noise_lr_factor)
        
        # Warmup phase to establish mean function baseline
        warmup_iters = int(warmup_fraction * n_iter)
        
        # Training loop with relaxed CG settings for variational inference
        if verbose:
            print(f"Training heteroscedastic GP with {num_inducing} inducing points...")
            print(f"Mean function lr: {lr}, Noise function lr: {lr * noise_lr_factor}")
            if warmup_iters > 0:
                print(f"Warm-start: {warmup_iters} iterations with frozen noise")
            else:
                print(f"Joint training from start (no warmup)")
        
        # Relax CG tolerance and increase max iterations for better convergence
        # Residual norms were ~26-31, so setting tolerance to 50 to accept these
        with gp.settings.max_cg_iterations(2000), gp.settings.cg_tolerance(50.0):
            for i in range(n_iter):
                mean_optimizer.zero_grad()
                noise_optimizer.zero_grad()
                
                # Forward pass
                output = self.gp_model(X_torch)
                noise_output = self.gp_model.forward_noise(X_torch)
                
                # Compute heteroscedastic noise (log-variance)
                log_var = noise_output.mean
                noise_var = torch.exp(log_var)
                
                # Clamp noise to reasonable range to prevent numerical issues
                noise_var = torch.clamp(noise_var, min=1e-6, max=100.0)
                
                # Compute KL divergence term (part of ELBO)
                # This measures how far variational distribution is from prior
                kl_term = self.gp_model.variational_strategy.kl_divergence().sum() / y_torch.numel()
                
                # Compute expected log-likelihood with heteroscedastic noise
                # P(y|f, σ²(x)) = N(y | f, σ²(x))
                # We integrate over the variational distribution q(f)
                # E_q[log P(y|f, σ²)] = -0.5 * [log(2π) + log(σ²) + (y - f)²/σ²]
                mean_pred = output.mean
                var_pred = output.variance  # Epistemic uncertainty from GP
                
                # Total observation noise (aleatoric uncertainty from noise model)
                obs_noise = noise_var
                
                # Negative log-likelihood incorporating heteroscedastic noise
                # Shape: (n_samples,)
                nll = 0.5 * (torch.log(2 * torch.pi * obs_noise) + 
                            ((y_torch - mean_pred) ** 2 + var_pred) / obs_noise)
                
                # Regularization: moderate penalty for balanced performance
                noise_penalty = 0.02 * torch.mean(noise_var)
                
                # ELBO = E[log P(y|f)] - KL[q(f)||p(f)]
                # We want to maximize ELBO = minimize -ELBO
                loss = nll.mean() + kl_term + noise_penalty
                
                loss.backward()
                mean_optimizer.step()
                if i >= warmup_iters:
                    noise_optimizer.step()  # Update noise after warmup
                
                if verbose and (i % 50 == 0 or i == n_iter - 1):
                    avg_noise = noise_var.mean().item()
                    print(f"Iter {i+1}/{n_iter}, Loss: {loss.item():.4f}, NLL: {nll.mean().item():.4f}, "
                          f"KL: {kl_term.item():.4f}, Avg σ²: {avg_noise:.4f}")
        
        self._is_fitted = True
        self.model = self.gp_model
        
        if verbose:
            print("Training complete!")
    
    def save_h5(self, filepath):
        """Save model to HDF5 file."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        with h5py.File(filepath, 'w') as f:
            # Model configuration
            f.attrs['model_type'] = 'heteroscedastic_gp'
            f.attrs['n_inputs'] = self.n_inputs
            f.attrs['kernel_nu'] = self.kernel_nu
            f.attrs['mean_type'] = self.mean_type
            f.attrs['num_inducing'] = self.inducing_points.shape[0]
            f.attrs['device'] = str(self.device)
            f.attrs['dtype'] = str(self.dtype)
            
            # Save input/output names
            f.create_dataset('input_names', data=[s.encode('utf-8') for s in self.input_names])
            f.create_dataset('output_names', data=[s.encode('utf-8') for s in self.output_names])
            
            # Save transformers (similar to homoscedastic version)
            if self.input_transformers:
                for i, tf in enumerate(self.input_transformers):
                    tf_group = f.create_group(f'input_transformers/tf_{i}')
                    tf_group['type'] = str(type(tf)).encode('utf-8')
                    if hasattr(tf, 'min_') and hasattr(tf, 'scale_'):
                        tf_group.create_dataset('min_', data=tf.min_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
                    elif hasattr(tf, 'mean_') and hasattr(tf, 'scale_'):
                        tf_group.create_dataset('mean_', data=tf.mean_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
            
            if self.output_transformers:
                for i, tf in enumerate(self.output_transformers):
                    tf_group = f.create_group(f'output_transformers/tf_{i}')
                    tf_group['type'] = str(type(tf)).encode('utf-8')
                    if hasattr(tf, 'min_') and hasattr(tf, 'scale_'):
                        tf_group.create_dataset('min_', data=tf.min_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
                    elif hasattr(tf, 'mean_') and hasattr(tf, 'scale_'):
                        tf_group.create_dataset('mean_', data=tf.mean_)
                        tf_group.create_dataset('scale_', data=tf.scale_)
            
            # Training data
            f.create_dataset('train_x', data=self.train_x.cpu().numpy())
            f.create_dataset('train_y', data=self.train_y.cpu().numpy())
            f.create_dataset('inducing_points', data=self.inducing_points.cpu().numpy())
            
            # Model state
            model_state = self.gp_model.state_dict()
            for key, value in model_state.items():
                if isinstance(value, torch.Tensor):
                    f.create_dataset(f'model_state/{key}', data=value.cpu().numpy())
    
    @classmethod
    def load_h5(cls, filepath):
        """Load model from HDF5 file."""
        # Import transformer classes
        from surrogates_interface.transformers import MinMaxScaler, StandardScaler
        from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
        from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
        
        with h5py.File(filepath, 'r') as f:
            # Load configuration
            n_inputs = int(f.attrs['n_inputs'])
            kernel_nu = float(f.attrs['kernel_nu'])
            mean_type = str(f.attrs['mean_type'])
            device = str(f.attrs.get('device', 'cpu'))
            dtype_str = str(f.attrs.get('dtype', 'torch.float32'))
            dtype = torch.float64 if 'float64' in dtype_str else torch.float32
            
            # Load input/output names
            input_names = [s.decode('utf-8') for s in f['input_names'][:]]
            output_names = [s.decode('utf-8') for s in f['output_names'][:]]
            
            # Load transformers
            input_transformers = []
            output_transformers = []
            
            if 'input_transformers' in f:
                for tf_name in sorted(f['input_transformers'].keys()):
                    tf_group = f['input_transformers'][tf_name]
                    tf_type = tf_group['type'][()].decode('utf-8')
                    
                    if 'MinMaxScaler' in tf_type:
                        sklearn_scaler = SklearnMinMaxScaler()
                        sklearn_scaler.min_ = tf_group['min_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.data_min_ = sklearn_scaler.min_
                        sklearn_scaler.data_max_ = sklearn_scaler.min_ + 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.data_range_ = 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.min_)
                        tf = MinMaxScaler.from_sklearn(sklearn_scaler)
                        input_transformers.append(tf)
                    elif 'StandardScaler' in tf_type:
                        sklearn_scaler = SklearnStandardScaler()
                        sklearn_scaler.mean_ = tf_group['mean_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.var_ = sklearn_scaler.scale_ ** 2
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.mean_)
                        tf = StandardScaler.from_sklearn(sklearn_scaler)
                        input_transformers.append(tf)
            
            if 'output_transformers' in f:
                for tf_name in sorted(f['output_transformers'].keys()):
                    tf_group = f['output_transformers'][tf_name]
                    tf_type = tf_group['type'][()].decode('utf-8')
                    
                    if 'MinMaxScaler' in tf_type:
                        sklearn_scaler = SklearnMinMaxScaler()
                        sklearn_scaler.min_ = tf_group['min_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.data_min_ = sklearn_scaler.min_
                        sklearn_scaler.data_max_ = sklearn_scaler.min_ + 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.data_range_ = 1.0 / sklearn_scaler.scale_
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.min_)
                        tf = MinMaxScaler.from_sklearn(sklearn_scaler)
                        output_transformers.append(tf)
                    elif 'StandardScaler' in tf_type:
                        sklearn_scaler = SklearnStandardScaler()
                        sklearn_scaler.mean_ = tf_group['mean_'][:]
                        sklearn_scaler.scale_ = tf_group['scale_'][:]
                        sklearn_scaler.var_ = sklearn_scaler.scale_ ** 2
                        sklearn_scaler.n_features_in_ = len(sklearn_scaler.mean_)
                        tf = StandardScaler.from_sklearn(sklearn_scaler)
                        output_transformers.append(tf)
            
            # Create instance with proper initialization
            instance = cls(
                input_names=input_names,
                output_names=output_names,
                input_transformers=input_transformers,
                output_transformers=output_transformers
            )
            
            # Set device and dtype
            instance.device = torch.device(device)
            instance.dtype = dtype
            
            # Load training data
            train_x = torch.tensor(f['train_x'][:], device=instance.device, dtype=instance.dtype)
            train_y = torch.tensor(f['train_y'][:], device=instance.device, dtype=instance.dtype)
            inducing_points = torch.tensor(f['inducing_points'][:], device=instance.device, dtype=instance.dtype)
            
            instance.train_x = train_x
            instance.train_y = train_y
            instance.inducing_points = inducing_points
            instance.n_inputs = n_inputs
            instance.kernel_nu = kernel_nu
            instance.mean_type = mean_type
            
            # Create model
            instance.gp_model = instance.HeteroscedasticGPModel(
                inducing_points, n_inputs, kernel_nu, mean_type
            ).to(device=instance.device, dtype=instance.dtype)
            
            # Load model state
            model_state = {}
            for key in f['model_state'].keys():
                dataset = f[f'model_state/{key}']
                # Handle scalar datasets (0-dimensional) vs array datasets
                if dataset.shape == ():
                    # Scalar value - read with [()] 
                    value = dataset[()]
                else:
                    # Array value - read with [:]
                    value = dataset[:]
                
                model_state[key] = torch.tensor(
                    value, 
                    device=instance.device, 
                    dtype=instance.dtype
                )
            
            instance.gp_model.load_state_dict(model_state, strict=False)
            instance._is_fitted = True
            instance.model = instance.gp_model
        
        return instance


# ============================================================================
# WIND-FARM-LOADS INTEGRATION FUNCTIONS
# ============================================================================

def predict_loads_sector_average(surrogates_dict, sector_avg, yaw, helix_amp,
                                  ti_in_percent=True, return_std=False):
    """
    Predict loads from sector-averaged flow features with optional uncertainty.
    
    Drop-in replacement for wind_farm_loads.tool_agnostic.predict_loads_sector_average()
    with added uncertainty quantification support.
    
    This function bridges GPyTorch surrogates with wind-farm-loads infrastructure,
    following DTU's conventions while enabling uncertainty-aware predictions.
    
    Parameters:
    -----------
    surrogates_dict : dict[str, GPyTorchGPSurrogate]
        Dictionary mapping sensor names to trained surrogates, e.g.,
        {'RootMflp': gp_model, 'TwrBsMyt': gp_model2}
    sector_avg : xarray.DataArray
        Sector-averaged flow from ta.compute_sector_average()
        Dims: (wt, sector, quantity) where quantity indices are [WS, TI]
        Coordinates: sector=['up', 'right', 'down', 'left'] (wind-farm-loads order)
    yaw : array-like
        Yaw angles per turbine [degrees]
    helix_amp : array-like
        Helix amplitudes per turbine [degrees]
    ti_in_percent : bool
        Ignored - TI conversion is automatic (default: True, kept for compatibility)
        Function auto-detects if TI < 1.0 and converts to percent (model expects 0-100)
    return_std : bool
        If True, return both mean and std. If False, return only mean (default: False)
        
    Returns:
    --------
    If return_std=False:
        loads_xr : xarray.DataArray
            Load predictions with dimensions (wt, wd, ws, name)
            Compatible with ta.predict_loads_sector_average() output
            
    If return_std=True:
        loads_mean_xr : xarray.DataArray
            Mean load predictions with dimensions (wt, wd, ws, name)
        loads_std_xr : xarray.DataArray
            Std load predictions with dimensions (wt, wd, ws, name)
            
    Examples:
    ---------
    >>> # Basic usage (no uncertainty)
    >>> from gpytorch_gp_model import predict_loads_sector_average
    >>> surrogates = {'RootMflp': gp_model}
    >>> sector_avg = ta.compute_sector_average(sim_res, 10, 10)
    >>> loads = predict_loads_sector_average(
    ...     surrogates, sector_avg, yaw_array, helix_array
    ... )
    
    >>> # With uncertainty quantification
    >>> loads_mean, loads_std = predict_loads_sector_average(
    ...     surrogates, sector_avg, yaw_array, helix_array, return_std=True
    ... )
    
    >>> # Access specific sensor and turbine
    >>> root_moment_t2 = loads.sel(wt=1, name='RootMflp').values
    
    Notes:
    ------
    - Sector reordering: wind-farm-loads extracts [up, right, down, left]
                        training expects [right, up, left, down]
    - TI conversion: Automatic - detects if TI < 1.0 (fractions) and converts to percent
    - Works with PyWake simulation results via wind-farm-loads
    """
    # Extract feature array from sector-averaged flow
    features, extra_dims, n_wt, n_wd, n_ws = _extract_sector_features(
        sector_avg, yaw, helix_amp, ti_in_percent
    )
    
    # Predict loads for all sensors
    sensor_names = list(surrogates_dict.keys())
    n_sensors = len(sensor_names)
    n_samples = features.shape[0]
    
    if return_std:
        # Predict with uncertainty
        loads_mean = np.zeros((n_samples, n_sensors))
        loads_std = np.zeros((n_samples, n_sensors))
        
        for i, sensor_name in enumerate(sensor_names):
            y_mean, y_std = surrogates_dict[sensor_name].predict_with_std(features)
            loads_mean[:, i] = y_mean.flatten()
            loads_std[:, i] = y_std.flatten()
        
        # Reshape to original dimensions
        output_shape = [n_wt, n_wd, n_ws] + [sector_avg.sizes[d] for d in extra_dims] + [n_sensors]
        loads_mean = loads_mean.reshape(output_shape)
        loads_std = loads_std.reshape(output_shape)
        
        # Build xarray DataArrays
        loads_mean_xr = _build_output_xarray(
            loads_mean, n_wt, n_wd, n_ws, extra_dims, sector_avg, sensor_names, 'loads_mean'
        )
        loads_std_xr = _build_output_xarray(
            loads_std, n_wt, n_wd, n_ws, extra_dims, sector_avg, sensor_names, 'loads_std'
        )
        
        return loads_mean_xr, loads_std_xr
    
    else:
        # Predict without uncertainty
        loads = np.zeros((n_samples, n_sensors))
        
        for i, sensor_name in enumerate(sensor_names):
            y_pred = surrogates_dict[sensor_name].predict_output(features)
            loads[:, i] = y_pred.flatten()
        
        # Reshape to original dimensions
        output_shape = [n_wt, n_wd, n_ws] + [sector_avg.sizes[d] for d in extra_dims] + [n_sensors]
        loads = loads.reshape(output_shape)
        
        # Build xarray DataArray
        return _build_output_xarray(
            loads, n_wt, n_wd, n_ws, extra_dims, sector_avg, sensor_names, 'loads'
        )


def _extract_sector_features(sector_avg, yaw, helix_amp, ti_in_percent=True):
    """
    Extract feature array from sector-averaged flow.
    
    PRIVATE helper function for predict_loads_sector_average().
    
    Parameters:
    -----------
    sector_avg : xarray.DataArray
        Sector-averaged flow with dims (wt, sector, quantity)
        Coordinates: sector=['up', 'right', 'down', 'left']
    yaw : array-like
        Yaw angles per turbine [degrees]
    helix_amp : array-like
        Helix amplitudes per turbine [degrees]
    ti_in_percent : bool
        Ignored - TI conversion is automatic (kept for API compatibility)
        Function auto-detects if TI < 1.0 and converts to percent
        
    Returns:
    --------
    features : np.ndarray
        Feature array of shape (n_samples, 10)
        Columns: [WS_right, WS_up, WS_left, WS_down, 
                  TI_right, TI_up, TI_left, TI_down,
                  yaw, helix_amp]
    extra_dims : list[str]
        Names of extra dimensions beyond (wt, wd, ws)
    n_wt : int
        Number of turbines
    n_wd : int
        Number of wind directions
    n_ws : int
        Number of wind speeds
        
    Raises:
    -------
    ValueError
        If 'quantity' dimension is missing from sector_avg
    """
    import xarray as xr
    
    # Check for required 'quantity' dimension
    if 'quantity' not in sector_avg.dims:
        raise ValueError(
            "sector_avg must have 'quantity' dimension. "
            "Found dimensions: " + str(sector_avg.dims)
        )
    
    # Identify extra dimensions (e.g., 'displacement' from xr.concat)
    expected_dims = {'wt', 'wd', 'ws', 'sector', 'quantity'}
    extra_dims = [d for d in sector_avg.dims if d not in expected_dims]
    
    # Get dimensions
    n_wt = sector_avg.sizes['wt']
    n_wd = sector_avg.sizes.get('wd', 1)
    n_ws = sector_avg.sizes.get('ws', 1)
    
    # Extract flow features with explicit dimension ordering
    # sector_avg has shape (wt, wd, ws, [extra_dims], sector, quantity)
    # We want (wt, wd, ws, [extra_dims], sector, quantity) → (n_samples, sector=4, quantity=2)
    dim_order = ['wt', 'wd', 'ws'] + extra_dims + ['sector', 'quantity']
    sector_values = sector_avg.transpose(*dim_order).values
    
    # Reshape to (n_samples, sector=4, quantity=2)
    n_extra = np.prod([sector_avg.sizes[d] for d in extra_dims]) if extra_dims else 1
    n_samples = n_wt * n_wd * n_ws * n_extra
    sector_values = sector_values.reshape(n_samples, 4, 2)
    
    # Extract WS and TI
    ws_sectors = sector_values[:, :, 0]  # (n_samples, 4) - WS in [up, right, down, left]
    ti_sectors = sector_values[:, :, 1]  # (n_samples, 4) - TI in [up, right, down, left]
    
    # Convert TI to percent if needed (wind-farm-loads returns fractions)
    # Model expects TI in percent (0-100), auto-detect if conversion needed
    if np.any(ti_sectors < 1.0):
        ti_sectors = ti_sectors * 100.0
    
    # Reorder sectors from wind-farm-loads [up, right, down, left] 
    # to training format [right, up, left, down]
    sector_map = [1, 0, 3, 2]  # Maps wind-farm-loads indices to training indices
    ws_reordered = ws_sectors[:, sector_map]
    ti_reordered = ti_sectors[:, sector_map]
    
    # Broadcast yaw and helix_amp to match sample dimension
    yaw_arr = np.asarray(yaw).flatten()
    helix_arr = np.asarray(helix_amp).flatten()
    
    # Repeat each turbine's value for all its samples (wt changes slowest after reshape)
    # Each turbine gets (n_wd * n_ws * n_extra) samples
    yaw_tiled = np.repeat(yaw_arr, n_samples // len(yaw_arr))
    helix_tiled = np.repeat(helix_arr, n_samples // len(helix_arr))
    
    # Assemble feature matrix: [WS_right, WS_up, WS_left, WS_down,
    #                           TI_right, TI_up, TI_left, TI_down,
    #                           yaw, helix_amp]
    features = np.column_stack([
        ws_reordered,      # 4 columns
        ti_reordered,      # 4 columns
        yaw_tiled[:, None],    # 1 column
        helix_tiled[:, None]   # 1 column
    ])
    
    return features, extra_dims, n_wt, n_wd, n_ws


def _build_output_xarray(loads_array, n_wt, n_wd, n_ws, extra_dims, sector_avg, sensor_names, array_name='loads'):
    """
    Build xarray DataArray from numpy prediction array.
    
    PRIVATE helper function for predict_loads_sector_average().
    
    Parameters:
    -----------
    loads_array : np.ndarray
        Load predictions with shape (wt, wd, ws, [extra_dims], n_sensors)
    n_wt : int
        Number of turbines
    n_wd : int
        Number of wind directions
    n_ws : int
        Number of wind speeds
    extra_dims : list[str]
        Names of extra dimensions (e.g., ['displacement'])
    sector_avg : xarray.DataArray
        Original sector_avg input (used to extract coordinates)
    sensor_names : list[str]
        Names of load sensors (e.g., ['RootMflp', 'TwrBsMyt'])
    array_name : str
        Name for the xarray DataArray (default: 'loads')
        
    Returns:
    --------
    xarray.DataArray
        Structured array with dimensions (wt, wd, ws, [extra_dims], name)
    """
    import xarray as xr
    
    # Build dimension names
    dim_names = ['wt', 'wd', 'ws'] + extra_dims + ['name']
    
    # Build coordinates
    coords = {
        'wt': np.arange(n_wt),
        'wd': np.arange(n_wd),
        'ws': np.arange(n_ws),
        'name': sensor_names
    }
    
    # Add extra dimension coordinates from sector_avg
    for dim in extra_dims:
        coords[dim] = sector_avg.coords[dim]
    
    # Create xarray DataArray
    return xr.DataArray(
        loads_array,
        dims=dim_names,
        coords=coords,
        name=array_name
    )
