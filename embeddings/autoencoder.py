import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, lr=0.001):
        """
        input_dim: d (original space R^d)
        hidden_dims: list of hidden layer sizes
        latent_dim: k (latent space R^k, where k << d)
        """
        self.lr = lr
        
        # Build encoder layer dims: d -> hidden -> k
        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        # Build decoder layer dims: k -> hidden (reversed) -> d
        dec_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        # Initialize weights (He initialization, good for ReLU)
        self.enc_W = [np.random.randn(enc_dims[i+1], enc_dims[i]) * np.sqrt(2/enc_dims[i]) 
                      for i in range(len(enc_dims)-1)]
        self.enc_b = [np.zeros((enc_dims[i+1], 1)) 
                      for i in range(len(enc_dims)-1)]
        
        self.dec_W = [np.random.randn(dec_dims[i+1], dec_dims[i]) * np.sqrt(2/dec_dims[i]) 
                      for i in range(len(dec_dims)-1)]
        self.dec_b = [np.zeros((dec_dims[i+1], 1)) 
                      for i in range(len(dec_dims)-1)]

    # forward pass activations
    def relu(self, a):
        return np.maximum(0, a)

    def relu_grad(self, a):
        # Sub-derivative: 0 if a < 0, 1 if a > 0, any value in [0,1] at a=0
        return (a > 0).astype(float)

    # ---------- Forward pass ----------
    def encode(self, x):
        """
        Implements F(x) = U_L * ReLU(U_{L-1} * ReLU(... U_0*x + b_0 ...) + b_{L-1}) + b_L
        Returns latent z and cache of pre/post activations for backprop
        """
        cache = []
        h = x
        for i, (W, b) in enumerate(zip(self.enc_W, self.enc_b)):
            pre = W @ h + b                          # linear transform
            post = self.relu(pre) if i < len(self.enc_W)-1 else pre  # no ReLU on last layer
            cache.append((h, pre))
            h = post
        return h, cache  # h = z (latent vector)

    def decode(self, z):
        """
        Implements G(z) = W_L * ReLU(W_{L-1} * ReLU(... W_0*z + c_0 ...) + c_{L-1}) + c_L
        Returns reconstruction x_hat and cache
        """
        cache = []
        h = z
        for i, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
            pre = W @ h + b
            post = self.relu(pre) if i < len(self.dec_W)-1 else pre  # no ReLU on last layer
            cache.append((h, pre))
            h = post
        return h, cache  # h = x_hat

    def forward(self, x):
        z, enc_cache = self.encode(x)
        x_hat, dec_cache = self.decode(z)
        return x_hat, z, enc_cache, dec_cache

    # loss function
    def mse_loss(self, x, x_hat):
        # (1/N) * sum ||x_n - G(F(x_n))||^2  — from your professor's objective
        return np.mean(np.sum((x - x_hat)**2, axis=0))

    # backwards pass
    def backward(self, x, x_hat, enc_cache, dec_cache):
        N = x.shape[1]

        # Gradient of MSE loss w.r.t. x_hat: dL/dx_hat = -(2/N)(x - x_hat)
        grad = -(2 / N) * (x - x_hat)

        # decoder backprop
        dec_dW, dec_db = [], []
        for i in reversed(range(len(self.dec_W))):
            h_prev, pre = dec_cache[i]
            # Apply ReLU gradient (skip on last layer — no activation there)
            if i < len(self.dec_W) - 1:
                grad = grad * self.relu_grad(pre)
            dW = grad @ h_prev.T
            db = np.sum(grad, axis=1, keepdims=True)
            grad = self.dec_W[i].T @ grad   # pass gradient back through weights
            dec_dW.insert(0, dW)
            dec_db.insert(0, db)

        # grad is now dL/dz — passes into encoder
        # encoder backprop
        enc_dW, enc_db = [], []
        for i in reversed(range(len(self.enc_W))):
            h_prev, pre = enc_cache[i]
            if i < len(self.enc_W) - 1:
                grad = grad * self.relu_grad(pre)
            dW = grad @ h_prev.T
            db = np.sum(grad, axis=1, keepdims=True)
            grad = self.enc_W[i].T @ grad
            enc_dW.insert(0, dW)
            enc_db.insert(0, db)

        return enc_dW, enc_db, dec_dW, dec_db

    # update
    def step(self, enc_dW, enc_db, dec_dW, dec_db):
        for i in range(len(self.enc_W)):
            self.enc_W[i] -= self.lr * enc_dW[i]
            self.enc_b[i] -= self.lr * enc_db[i]
        for i in range(len(self.dec_W)):
            self.dec_W[i] -= self.lr * dec_dW[i]
            self.dec_b[i] -= self.lr * dec_db[i]

    # training loop
    def train(self, X, epochs=500, batch_size=64):
        """
        X: shape (d, N) — each column is one sample
        """
        N = X.shape[1]
        for epoch in range(epochs):
            # shuffle
            idx = np.random.permutation(N)
            X_shuffled = X[:, idx]
            
            for i in range(0, N, batch_size):
                batch_X = X_shuffled[:, i:i+batch_size]
                x_hat, z, enc_cache, dec_cache = self.forward(batch_X)
                enc_dW, enc_db, dec_dW, dec_db = self.backward(batch_X, x_hat, enc_cache, dec_cache)
                self.step(enc_dW, enc_db, dec_dW, dec_db)
