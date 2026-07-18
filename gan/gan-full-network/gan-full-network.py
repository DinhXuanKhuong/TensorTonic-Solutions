import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)

        
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        # pass
        return np.round(np.tanh(z @ self.G_W), 4)
        
    def sigmoid(self, x):
        return np.round(1.0 / (1.0 + np.exp(-x)), 4)
        
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        # pass
        return self.sigmoid(x @ self.D_W)
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        # Your implementation here
        # pass
        fake_data = self.generate(z)
        p_real = self.discriminate(real_data)
        p_fake = self.discriminate(fake_data)
        # p_fake = self.generate(z)

        d_loss = -np.mean(np.log(p_real) + np.log(1 - p_fake))
    
        g_loss = -np.mean(np.log(p_fake))
    
        return {"d_loss" : np.round(d_loss, 4),
               "g_loss": np.round(g_loss, 4)}
        

        