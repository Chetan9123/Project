from PIL import Image, ImageDraw
import os

def create_default_avatar():
    # Create a 200x200 image with a light gray background
    size = 200
    image = Image.new('RGB', (size, size), '#f0f0f0')
    draw = ImageDraw.Draw(image)
    
    # Draw a simple user icon
    # Head
    head_radius = 60
    head_center = (size//2, size//2 - 20)
    draw.ellipse(
        [
            head_center[0] - head_radius,
            head_center[1] - head_radius,
            head_center[0] + head_radius,
            head_center[1] + head_radius
        ],
        fill='#808080'
    )
    
    # Body
    body_top = head_center[1] + head_radius
    body_bottom = size - 40
    body_width = 80
    draw.rectangle(
        [
            size//2 - body_width//2,
            body_top,
            size//2 + body_width//2,
            body_bottom
        ],
        fill='#808080'
    )
    
    # Save the image
    os.makedirs('static/img', exist_ok=True)
    image.save('static/img/default-avatar.png')

if __name__ == '__main__':
    create_default_avatar() 