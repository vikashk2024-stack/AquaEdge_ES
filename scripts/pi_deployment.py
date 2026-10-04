import time
from collections import Counter
from ultralytics import YOLO

# ---------------------------------------------------------
# Raspberry Pi Hardware Setup (Uncomment when on the Pi!)
# ---------------------------------------------------------
# import RPi.GPIO as GPIO
# from RPLCD.i2c import CharLCD
# 
# # Setup LCD (Common I2C address is 0x27)
# lcd = CharLCD('PCF8574', 0x27)
#
# # Setup Alert LED (Example on GPIO pin 18)
# LED_PIN = 18
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(LED_PIN, GPIO.OUT)

def update_lcd_display(text_line_1, text_line_2):
    """Helper function to write to the LCD display"""
    print(f"LCD Line 1: {text_line_1}")
    print(f"LCD Line 2: {text_line_2}")
    
    # Uncomment when running on Raspberry Pi:
    # lcd.clear()
    # lcd.write_string(text_line_1 + '\r\n' + text_line_2)

def flash_led_alert():
    """Flash LED if dangerous microorganisms are found"""
    print("WARNING: Flashing LED!")
    # Uncomment when running on Raspberry Pi:
    # GPIO.output(LED_PIN, GPIO.HIGH)
    # time.sleep(1)
    # GPIO.output(LED_PIN, GPIO.LOW)


# ---------------------------------------------------------
# AquaEdge YOLOv11 Model Integration
# ---------------------------------------------------------

# Path to the model you will copy over to the Raspberry Pi
MODEL_PATH = "best.pt" 

try:
    model = YOLO(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Could not find '{MODEL_PATH}'. Make sure you copy your trained model to the Pi!")
    exit(1)

def analyze_water_sample(image_path):
    print(f"\nAnalyzing sample: {image_path}...")
    
    # Run YOLO with a confidence threshold
    results = model(image_path, conf=0.25)
    
    # Extract detected classes
    detected_classes = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    
    if len(detected_classes) == 0:
        update_lcd_display("Water is clear!", "No organisms.")
        return
    
    # Count occurrences of each microorganism
    counts = Counter(detected_classes)
    
    # Format the most common microorganism for the 16x2 LCD display
    most_common_id, top_count = counts.most_common(1)[0]
    top_name = class_names[int(most_common_id)]
    
    # Example LCD Output: 
    # "Detected: 12"
    # "Main: Chlorella"
    total_found = len(detected_classes)
    line1 = f"Detected: {total_found}"
    line2 = f"Main: {top_name}"
    
    update_lcd_display(line1, line2)
    
    # Example Hardware Logic: If a dangerous algae is found (like Anabaena), flash an LED!
    # (Assuming Anabaena is a class you want to flag)
    if "Anabaena" in [class_names[int(c)] for c in detected_classes]:
        flash_led_alert()

if __name__ == "__main__":
    # Example usage: Test with a sample image
    # (On the actual Pi, you can replace this with capturing a photo from the Pi Camera)
    test_image = "sample_water.jpg"
    print("Starting AquaEdge Pi Deployment Script...")
    
    # You would typically run this in a loop capturing from the camera
    # analyze_water_sample(test_image)
    print("\nScript ready! Copy 'best.pt' and this script to your Raspberry Pi to get started.")
