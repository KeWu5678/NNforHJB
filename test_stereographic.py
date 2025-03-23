import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereo(z):
    """
    Standard stereographic projection function from your code.
    z shape: (2, n)
    returns: a (2, n), b (1, n)
    """
    denominator = (1 + np.sum(z**2, axis=0)).reshape(1, -1)
    a = (2 * z) / denominator
    b = (1 - np.sum(z**2, axis=0)) / denominator
    return a, b

def inverse_stereo(a, b):
    """
    Inverse stereographic projection: Map from the unit ball (a, b) back to R^2 (z)
    a shape: (2, n)
    b shape: (1, n)
    returns: z (2, n)
    """
    b = b.reshape(1, -1)
    return a / (1 + b)

def calculate_activation_stats(X_train, z_samples, num_samples=1000):
    """
    Calculate statistics of activations for different z values
    """
    activation_positive_rates = []
    a_norms = []
    b_values = []
    
    for i in range(num_samples):
        z = z_samples[:, i:i+1]  # Get one sample as (2, 1)
        a, b = stereo(z)
        
        # Calculate activation
        activations = X_train @ a + b
        
        # Calculate statistics
        positive_rate = np.mean(activations > 0)
        activation_positive_rates.append(positive_rate)
        
        # Calculate norm of a (should be less than 1 for unit ball)
        a_norm = np.linalg.norm(a)
        a_norms.append(a_norm)
        
        # Store b value
        b_values.append(b[0, 0])
    
    return activation_positive_rates, a_norms, b_values

def plot_projected_points(a_samples, b_samples):
    """
    Plot the projected points (a, b) on a 3D unit sphere
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to 3D coordinates on unit sphere
    x = a_samples[0, :]
    y = a_samples[1, :]
    z = b_samples[0, :]
    
    # Plot points
    ax.scatter(x, y, z, c='b', marker='o', alpha=0.6)
    
    # Plot unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 0.99 * np.outer(np.cos(u), np.sin(v))
    y = 0.99 * np.outer(np.sin(u), np.sin(v))
    z = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='r', alpha=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Stereographic Projection on Unit Sphere')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    return fig

def test_sampling_methods():
    # Generate random training data
    np.random.seed(42)
    K = 100
    X_train = np.random.uniform(-1, 1, (K, 2))
    
    num_samples = 1000
    
    # Test different z sampling strategies
    print("Testing different sampling strategies for z...")
    
    # Method 1: Uniform sampling in a square [-r, r]²
    max_range = 5
    z_uniform_square = np.random.uniform(-max_range, max_range, (2, num_samples))
    a_uniform_square, b_uniform_square = stereo(z_uniform_square)
    
    # Method 2: Uniform sampling in a circle with radius r
    radius = 5
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    r = radius * np.sqrt(np.random.uniform(0, 1, num_samples))  # Square root for uniform distribution
    z_uniform_circle = np.vstack([r * np.cos(theta), r * np.sin(theta)])
    a_uniform_circle, b_uniform_circle = stereo(z_uniform_circle)
    
    # Method 3: Normal distribution
    z_normal = np.random.normal(0, 2, (2, num_samples))
    a_normal, b_normal = stereo(z_normal)
    
    # Method 4: Inverse sampling from uniform on sphere
    # Sample uniformly on unit sphere and then transform back to z-space
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    phi = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1)  # For uniform distribution on sphere
    
    # Convert to 3D Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Convert to a, b representation
    a_inverse = np.vstack([x, y])
    b_inverse = z.reshape(1, -1)
    
    # Get z using inverse projection
    z_inverse = inverse_stereo(a_inverse, b_inverse)
    
    # Verify with forward projection
    a_inverse_check, b_inverse_check = stereo(z_inverse)
    
    # Calculate activation statistics for each method
    print("\nCalculating activation statistics...")
    act_pos_uniform_square, a_norms_uniform_square, b_vals_uniform_square = calculate_activation_stats(X_train, z_uniform_square)
    act_pos_uniform_circle, a_norms_uniform_circle, b_vals_uniform_circle = calculate_activation_stats(X_train, z_uniform_circle)
    act_pos_normal, a_norms_normal, b_vals_normal = calculate_activation_stats(X_train, z_normal)
    act_pos_inverse, a_norms_inverse, b_vals_inverse = calculate_activation_stats(X_train, z_inverse)
    
    # Print statistics
    print("\nSampling Statistics:")
    print(f"Uniform Square: Avg positive activation rate = {np.mean(act_pos_uniform_square):.4f}, Avg a norm = {np.mean(a_norms_uniform_square):.4f}, Avg b = {np.mean(b_vals_uniform_square):.4f}")
    print(f"Uniform Circle: Avg positive activation rate = {np.mean(act_pos_uniform_circle):.4f}, Avg a norm = {np.mean(a_norms_uniform_circle):.4f}, Avg b = {np.mean(b_vals_uniform_circle):.4f}")
    print(f"Normal: Avg positive activation rate = {np.mean(act_pos_normal):.4f}, Avg a norm = {np.mean(a_norms_normal):.4f}, Avg b = {np.mean(b_vals_normal):.4f}")
    print(f"Inverse: Avg positive activation rate = {np.mean(act_pos_inverse):.4f}, Avg a norm = {np.mean(a_norms_inverse):.4f}, Avg b = {np.mean(b_vals_inverse):.4f}")
    
    # Distribution of b values
    print("\nDistribution of b values (quantiles):")
    for method_name, b_values in [
        ("Uniform Square", b_vals_uniform_square),
        ("Uniform Circle", b_vals_uniform_circle),
        ("Normal", b_vals_normal),
        ("Inverse", b_vals_inverse)
    ]:
        quantiles = np.quantile(b_values, [0, 0.25, 0.5, 0.75, 1])
        print(f"{method_name}: min={quantiles[0]:.4f}, Q1={quantiles[1]:.4f}, median={quantiles[2]:.4f}, Q3={quantiles[3]:.4f}, max={quantiles[4]:.4f}")
    
    # Check uniformity of distribution on the sphere for each method
    print("\nChecking uniformity of distribution on the unit sphere...")
    
    # Helper function to check distribution uniformity
    def check_uniformity(a_vals, b_vals, method_name):
        # Convert from a, b to 3D points
        points_3d = np.vstack([a_vals[0], a_vals[1], b_vals[0]])
        
        # Calculate octant counts (divide sphere into 8 parts)
        octants = [(points_3d[0] > 0) & (points_3d[1] > 0) & (points_3d[2] > 0),
                   (points_3d[0] < 0) & (points_3d[1] > 0) & (points_3d[2] > 0),
                   (points_3d[0] > 0) & (points_3d[1] < 0) & (points_3d[2] > 0),
                   (points_3d[0] < 0) & (points_3d[1] < 0) & (points_3d[2] > 0),
                   (points_3d[0] > 0) & (points_3d[1] > 0) & (points_3d[2] < 0),
                   (points_3d[0] < 0) & (points_3d[1] > 0) & (points_3d[2] < 0),
                   (points_3d[0] > 0) & (points_3d[1] < 0) & (points_3d[2] < 0),
                   (points_3d[0] < 0) & (points_3d[1] < 0) & (points_3d[2] < 0)]
        
        octant_counts = [np.sum(oct) for oct in octants]
        
        # Calculate the standard deviation of counts (smaller = more uniform)
        std_dev = np.std(octant_counts)
        total = np.sum(octant_counts)
        expected = total / 8
        
        print(f"{method_name}: Octant distribution std dev = {std_dev:.2f} (Ideal = 0)")
        for i, count in enumerate(octant_counts):
            print(f"  Octant {i+1}: {count} points ({count/total*100:.1f}%, Expected: {expected:.1f})")
    
    check_uniformity(a_uniform_square, b_uniform_square, "Uniform Square")
    check_uniformity(a_uniform_circle, b_uniform_circle, "Uniform Circle")
    check_uniformity(a_normal, b_normal, "Normal")
    check_uniformity(a_inverse, b_inverse, "Inverse")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("Based on the analysis, here's how to sample z for uniform distribution on the unit ball:")
    print("1. For truly uniform distribution on the unit sphere: Use the inverse sampling method")
    print("   - Sample points uniformly on the unit sphere")
    print("   - Apply the inverse stereographic projection to get z values")
    print("2. For simple approach with good coverage: Use normal distribution with mean=0, std=2")
    print("   - Adjust the standard deviation to control the concentration of points")
    print("3. For balanced positive/negative activations: Use the method with activation rates closest to 0.5")
    
    # Create plots
    print("\nGenerating plots (these would typically be saved to disk)...")
    
    # If not in an environment with display capabilities, comment out the following plotting code
    
    # Plot projections
    fig1 = plot_projected_points(a_uniform_square, b_uniform_square)
    plt.title("Uniform Square Sampling Projection")
    plt.tight_layout()
    
    fig2 = plot_projected_points(a_uniform_circle, b_uniform_circle)
    plt.title("Uniform Circle Sampling Projection")
    plt.tight_layout()
    
    fig3 = plot_projected_points(a_normal, b_normal)
    plt.title("Normal Distribution Sampling Projection")
    plt.tight_layout()
    
    fig4 = plot_projected_points(a_inverse, b_inverse)
    plt.title("Inverse Sampling Projection (Most Uniform)")
    plt.tight_layout()
    
    # Display all plots (or save to files if running headless)
    plt.show()

if __name__ == "__main__":
    test_sampling_methods()
    
    # If the above fails due to matplotlib in your environment, run this simpler test
    try:
        test_sampling_methods()
    except ImportError:
        print("Matplotlib not available, running simplified test...")
        
        # Generate random training data
        np.random.seed(42)
        K = 100
        X_train = np.random.uniform(-1, 1, (K, 2))
        
        # Test just one method - inverse sampling for most uniform distribution
        num_samples = 1000
        
        # Sample uniformly on unit sphere and then transform back to z-space
        theta = np.random.uniform(0, 2*np.pi, num_samples)
        phi = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1)
        
        # Convert to 3D Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Convert to a, b representation (these are uniformly distributed on unit sphere)
        a_inverse = np.vstack([x, y])
        b_inverse = z.reshape(1, -1)
        
        # Get z using inverse projection
        z_inverse = inverse_stereo(a_inverse, b_inverse)
        
        # Verify and print some sample values
        for i in range(5):
            sample_z = z_inverse[:, i:i+1]
            sample_a, sample_b = stereo(sample_z)
            print(f"Sample {i+1}:")
            print(f"  z = {sample_z.flatten()}")
            print(f"  a = {sample_a.flatten()}, b = {sample_b.flatten()}")
            print(f"  ||(a,b)|| = {np.sqrt(np.sum(sample_a**2) + sample_b**2):.4f} (should be close to 1)")
        
        # Calculate activation stats
        act_pos, a_norms, b_vals = calculate_activation_stats(X_train, z_inverse)
        print(f"\nAvg positive activation rate: {np.mean(act_pos):.4f}")
        print(f"Min/Max b values: {np.min(b_vals):.4f} to {np.max(b_vals):.4f}")
        
        # Final recommendation
        print("\nRECOMMENDATION:")
        print("To get uniformly distributed points on the unit sphere via stereographic projection:")
        print("1. Generate uniform points on the unit sphere using:")
        print("   theta = np.random.uniform(0, 2*np.pi, num_samples)")
        print("   phi = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1)")
        print("   x = np.sin(phi) * np.cos(theta)")
        print("   y = np.sin(phi) * np.sin(theta)")
        print("   z = np.cos(phi)")
        print("2. Create a and b as:")
        print("   a = np.vstack([x, y])")
        print("   b = z.reshape(1, -1)")
        print("3. Apply inverse stereographic projection to get z:")
        print("   z = a / (1 + b)") 