import cv2
import torch
import numpy as np
import math

def calculate_volume(length_m, width_m, depth_m):
    """
    Estimate pothole volume in cubic meters using a rectangular approximation.
    """
    return length_m * width_m * depth_m

def estimate_cost(volume_m3, asphalt_cost_per_ton=5000, asphalt_density_ton_per_m3=2.4,
                  labor_cost=2000, equipment_cost=1500, other_overheads=1000):
    """
    Estimate the cost of pothole repair in INR.
    """
    # Material cost
    material_cost = volume_m3 * asphalt_density_ton_per_m3 * asphalt_cost_per_ton

    # Total cost
    total_cost = material_cost + labor_cost + equipment_cost + other_overheads

    return {
        "volume_m3": volume_m3,
        "material_cost": round(material_cost, 2),
        "labor_cost": labor_cost,
        "equipment_cost": equipment_cost,
        "other_overheads": other_overheads,
        "total_cost": round(total_cost, 2)
    }

def stereo_depth_estimation(left_img_path, right_img_path):
    """
    Estimate depth using stereo vision (two images).
    """
    # Load stereo images
    img_left = cv2.imread(left_img_path, 0)  # Left image
    img_right = cv2.imread(right_img_path, 0)  # Right image

    # Create StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    # Compute disparity map
    disparity = stereo.compute(img_left, img_right)

    # Return the depth map as disparity
    return disparity

def monocular_depth_estimation(image_path):
    """
    Estimate depth using MiDaS (monocular depth estimation).
    """
    # Load the MiDaS model (pre-trained)
    model_type = "DPT_Large"  # You can also try "DPT_Hybrid" or "MiDaS_v2_1"
    model = torch.hub.load("intel-isl/MiDaS", model_type)

    # Transform the input image
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

    # Read the image
    img = cv2.imread(image_path)

    # Preprocess the image
    input_image = transform(img).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Perform depth estimation
    with torch.no_grad():
        prediction = model(input_image)

    # Post-process the depth map
    prediction = prediction.squeeze().cpu().numpy()
    depth_map = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Return the depth map
    return depth_map

def main():
    # Get user input to select image mode (single or stereo)
    mode = input("Enter mode (1 for single image, 2 for stereo images): ")

    # If stereo images mode
    if mode == '2':
        left_image_path = input("Enter path to left image: ")
        right_image_path = input("Enter path to right image: ")

        # Calculate depth using stereo vision
        disparity = stereo_depth_estimation(left_image_path, right_image_path)

        # Find the average depth (simple average of disparity values for estimation)
        avg_disparity = np.mean(disparity)
        estimated_depth = 1 / avg_disparity  # Simplified inverse relationship (in real scenarios, use camera calibration)
        print(f"Estimated Depth (from stereo images): {estimated_depth:.2f} meters")

    # If single image mode
    elif mode == '1':
        image_path = input("Enter path to the image: ")

        # Calculate depth using monocular depth estimation
        depth_map = monocular_depth_estimation(image_path)

        # Visualize the depth map (optional)
        cv2.imshow('Depth Map', depth_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find the average depth from the depth map (this is a rough estimation)
        avg_depth = np.mean(depth_map) / 255.0  # Normalize depth (0 to 1 scale)
        print(f"Estimated Depth (from monocular image): {avg_depth:.2f} meters")

    else:
        print("Invalid mode selected!")

    # Now calculate volume and cost (you may want to ask the user for pothole dimensions)
    length = float(input("Enter length of pothole in meters: "))
    width = float(input("Enter width of pothole in meters: "))
    depth = float(input(f"Enter depth of pothole in meters (use {avg_depth:.2f} as estimate if unsure): "))

    # Calculate volume
    volume = calculate_volume(length, width, depth)
    cost_details = estimate_cost(volume)

    # Display cost estimation
    print("\n--- Pothole Repair Cost Estimate ---")
    print(f"Estimated Volume: {cost_details['volume_m3']:.2f} m³")
    print(f"Material Cost: ₹{cost_details['material_cost']}")
    print(f"Labor Cost: ₹{cost_details['labor_cost']}")
    print(f"Equipment Cost: ₹{cost_details['equipment_cost']}")
    print(f"Other Overheads: ₹{cost_details['other_overheads']}")
    print(f"✅ Total Estimated Cost: ₹{cost_details['total_cost']}")

if __name__ == "__main__":
    main()

