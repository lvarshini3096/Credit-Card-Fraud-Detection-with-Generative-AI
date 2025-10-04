# --- gan_model.py ---
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

# --- Model Hyperparameters ---
NOISE_DIM = 100
# Feature Dimension is 29 (30 original columns - 1 Class column)
FEATURE_DIM = 29 

def build_generator(noise_dim=NOISE_DIM, output_dim=FEATURE_DIM):
    """Creates the Generator model."""
    model = Sequential(name="Generator")
    init = RandomNormal(mean=0.0, stddev=0.02)
    
    # Layer 1
    model.add(Dense(128, kernel_initializer=init, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Layer 2
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    # Layer 3
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    # Output layer: 'tanh' to output between -1 and 1, matching scaled data
    model.add(Dense(output_dim, activation='tanh')) 
    return model

def build_discriminator(input_dim=FEATURE_DIM):
    """Creates the Discriminator model."""
    model = Sequential(name="Discriminator")
    
    # Layer 1
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Layer 2
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Output layer: 'sigmoid' for binary classification (Real: 1, Fake: 0)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the discriminator
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    model.trainable = True
    return model

def build_gan(generator, discriminator):
    """Combines Generator and Discriminator to form the GAN."""
    # Freeze discriminator for the combined model's training step
    discriminator.trainable = False 
    
    gan_input = Input(shape=(NOISE_DIM,))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)
    
    gan = Model(gan_input, gan_output, name="GAN")
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), 
                loss='binary_crossentropy')
    return gan

def train_gan(generator, discriminator, gan, real_data_df, epochs=10000, batch_size=256):
    """Custom training loop for the GAN."""
    print("Starting GAN training...")
    real_data = real_data_df.values
    
    for epoch in range(epochs):
        # --- 1. Train Discriminator (Maximize log(D(x)) + log(1 - D(G(z)))) ---
        
        # Get a random batch of real data
        idx = np.random.randint(0, real_data.shape[0], batch_size // 2)
        real_batch = real_data[idx]
        
        # Generate a batch of fake data
        noise = np.random.normal(0, 1, (batch_size // 2, NOISE_DIM))
        fake_batch = generator.predict(noise, verbose=0)
        
        # Combine and label (Real=1, Fake=0)
        X_combined = np.concatenate([real_batch, fake_batch])
        y_combined = np.array([1] * (batch_size // 2) + [0] * (batch_size // 2))
        
        d_loss, d_acc = discriminator.train_on_batch(X_combined, y_combined)
        
        # --- 2. Train Generator (Minimize log(1 - D(G(z))) or Maximize log(D(G(z)))) ---
        
        # Generate new noise batch
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        # Generator's target label is '1' (it wants to fool the discriminator)
        y_generator = np.array([1] * batch_size) 
        
        g_loss = gan.train_on_batch(noise, y_generator)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss:.4f} (Acc: {d_acc*100:.2f}%) | G Loss: {g_loss:.4f}")
            
    print("GAN training complete.")
    return generator

def generate_synthetic_data(generator, num_samples):
    """Generates the final synthetic dataset from the trained generator."""
    noise = np.random.normal(0, 1, (num_samples, NOISE_DIM))
    synthetic_data = generator.predict(noise, verbose=0)
    return synthetic_data

if __name__ == '__main__':
    from data_prep import load_and_preprocess_data
    
    # Load and preprocess the data
    X, y, df_fraud = load_and_preprocess_data()

    if X is not None and df_fraud is not None:
        # 1. Build Models
        generator = build_generator()
        discriminator = build_discriminator()
        gan = build_gan(generator, discriminator)

        # 2. Train the GAN using ONLY the real fraudulent data
        # Note: In the notebook, this part would have a much longer runtime.
        # Use fewer epochs for a quick test run.
        EPOCHS = 10000
        
        trained_generator = train_gan(generator, discriminator, gan, df_fraud, epochs=EPOCHS)
        
        # 3. Generate enough data to balance the dataset
        num_to_generate = X.shape[0] - df_fraud.shape[0]
        synthetic_fraud_data = generate_synthetic_data(trained_generator, num_to_generate)
        
        # 4. Create the final balanced dataset
        synthetic_df = pd.DataFrame(synthetic_fraud_data, columns=df_fraud.columns)
        synthetic_df['Class'] = 1
        
        df_genuine = X[y == 0].copy()
        df_genuine['Class'] = 0
        
        balanced_df = pd.concat([df_genuine, synthetic_df], ignore_index=True)

        print("\n--- Balanced Dataset Summary ---")
        print(balanced_df['Class'].value_counts())
        print(f"Balanced Dataset Shape: {balanced_df.shape}")

        # Optional: Train and evaluate a final LR on the balanced data
        # Note: This is an extra step for a complete solution.
        from lr_pca_model import train_and_evaluate_lr_pca
        X_bal = balanced_df.drop('Class', axis=1)
        y_bal = balanced_df['Class']
        
        print("\n--- Part 4: Training LR on GAN-Augmented Data ---")
        # Re-apply PCA/LR with a balanced split for better metrics
        train_and_evaluate_lr_pca(X_bal, y_bal, n_components=10)
