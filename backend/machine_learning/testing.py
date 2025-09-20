import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import os
import sys

img_size = (224, 224)
class_names = ['accident', 'non_accident']
FINAL_MODEL_PATH = r"D:\Smart Accident Detector\Updated frontend\Smart-Accident-Detector\machine_learning\accident_detection_model.h5"
BEST_MODEL_PATH = r"D:\Smart Accident Detector\Updated frontend\Smart-Accident-Detector\machine_learning\best_model.h5"

class AccidentDetector:
    def __init__(self):
        self.model_final = None
        self.model_best = None
        self.load_models()
    
    def load_models(self):
        try:
            print("Loading models...")
            self.model_final = tf.keras.models.load_model(FINAL_MODEL_PATH)
            self.model_best = tf.keras.models.load_model(BEST_MODEL_PATH)
            print("Both models loaded successfully!")
            print("Model output shape:", self.model_final.output_shape)
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path).resize(img_size)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array, img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None, None
    
    def predict_accident(self, image_path, model, model_name):
        img_array, original_img = self.preprocess_image(image_path)
        if img_array is None:
            return None, None, None, None
        
        raw_prediction = model.predict(img_array, verbose=0)[0][0]
        
        if raw_prediction < 0.5:
            predicted_class = 'accident'
            confidence = 1 - raw_prediction
            is_accident = True
        else:
            predicted_class = 'non_accident' 
            confidence = raw_prediction 
            is_accident = False
        
        print(f"DEBUG [{model_name}]: Raw prediction: {raw_prediction:.4f}, "
              f"Class: {predicted_class}, Confidence: {confidence:.4f}")
        
        return predicted_class, confidence, is_accident, original_img
    
    def get_location(self):
        try:
            geolocator = Nominatim(user_agent="accident_detector")
            location = geolocator.geocode("New York, NY, USA")
            return location.address if location else "Unknown location"
        except:
            return "Location unavailable"
    
    def display_prediction_results(self, image_path):
        print("\n" + "="*60)
        print("ANALYZING IMAGE...")
        print("="*60)
        
        pred_final, conf_final, is_acc_final, img = self.predict_accident(
            image_path, self.model_final, "Final Model"
        )
        pred_best, conf_best, is_acc_best, _ = self.predict_accident(
            image_path, self.model_best, "Best Model"
        )
        
        if pred_final is None:
            return
        
        print(f"PREDICTION RESULTS:")
        print(f"Final Model: {pred_final} (Confidence: {conf_final:.2%})")
        print(f"Best Model:  {pred_best} (Confidence: {conf_best:.2%})")
    
        confidence_threshold = 0.6 
        
        if (is_acc_final and conf_final > confidence_threshold) or \
           (is_acc_best and conf_best > confidence_threshold):
            print("\nACCIDENT DETECTED!")
            location = self.get_location()
            print(f"Location: {location}")
            print("Alert system would be triggered!")
        else:
            print("\nNo accident detected")
            if is_acc_final or is_acc_best:
                print("(Low confidence - not triggering alert)")
        
        self.visualize_prediction(img, pred_final, conf_final, pred_best, conf_best, image_path)
    
    def visualize_prediction(self, img, pred_final, conf_final, pred_best, conf_best, image_path):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, (1, 3))
        plt.imshow(img)
        plt.title(f"Input Image\n{os.path.basename(image_path)}", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        color_final = 'red' if pred_final == 'accident' else 'green'
        plt.bar(['Final Model'], [conf_final], color=color_final, alpha=0.7)
        plt.title(f'Final Model\n{pred_final}', fontweight='bold')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.text(0, conf_final + 0.05, f'{conf_final:.2%}', ha='center', fontweight='bold')
        
        plt.subplot(2, 2, 4)
        color_best = 'red' if pred_best == 'accident' else 'green'
        plt.bar(['Best Model'], [conf_best], color=color_best, alpha=0.7)
        plt.title(f'Best Model\n{pred_best}', fontweight='bold')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.text(0, conf_best + 0.05, f'{conf_best:.2%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def display_menu():
    print("\n" + "="*60)
    print("ACCIDENT DETECTION SYSTEM")
    print("="*60)
    print("1. Test single image")
    print("2. Exit")
    print("="*60)

def get_user_choice():
    while True:
        try:
            choice = input("\nEnter your choice (1-2): ").strip()
            if choice in ['1', '2']:
                return int(choice)
            else:
                print("Invalid choice. Please enter a number between 1-2.")
        except KeyboardInterrupt:
            print("\nTerminating the program...")
            sys.exit(0)

def main():
    print("Initializing Accident Detection System...")
    detector = AccidentDetector()
    
    while True:
        try:
            display_menu()
            choice = get_user_choice()
            
            if choice == 1:
                image_path = input("\nEnter image path: ").strip().strip('"\'')
                if os.path.exists(image_path):
                    detector.display_prediction_results(image_path)
                else:
                    print(f"Image not found: {image_path}")
            
            elif choice == 2:
                print("\nThank you for using Accident Detection System!")
                break
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\nTerminating...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accident Detection System")
    parser.add_argument('--image_path', type=str, help="Path to a single image (bypasses interactive mode)")
    
    args = parser.parse_args()
    
    if args.image_path:
        detector = AccidentDetector()
        detector.display_prediction_results(args.image_path)
    else:
        main()