#!/usr/bin/env python3
"""
Comprehensive Component Testing for BrainWave Analyzer
Tests all major components to ensure demo readiness
"""

import sys
import os
import time
import traceback
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComponentTester:
    """Comprehensive component testing suite"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        print(f"\n🧪 Testing {test_name}...")
        self.total_tests += 1
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                print(f"  ✅ PASSED ({duration:.2f}s)")
                self.test_results[test_name] = True
                self.passed_tests += 1
            else:
                print(f"  ❌ FAILED ({duration:.2f}s)")
                self.test_results[test_name] = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            print(f"  📋 Traceback: {traceback.format_exc()}")
            self.test_results[test_name] = False
    
    def test_imports(self):
        """Test basic imports"""
        try:
            # Test standard library imports
            import json, time, os, sys
            print("  ✅ Standard library imports")
            
            # Test scientific computing imports
            import numpy as np
            print("  ✅ NumPy import")
            
            # Test our custom modules (without dependencies)
            if Path('data/__init__.py').exists():
                print("  ✅ Data module structure")
            
            if Path('models/__init__.py').exists():
                print("  ✅ Models module structure")
            
            if Path('training/__init__.py').exists():
                print("  ✅ Training module structure")
            
            if Path('evaluation/__init__.py').exists():
                print("  ✅ Evaluation module structure")
            
            if Path('utils/__init__.py').exists():
                print("  ✅ Utils module structure")
            
            return True
        except Exception as e:
            print(f"  ❌ Import error: {e}")
            return False
    
    def test_data_generation(self):
        """Test synthetic data generation"""
        try:
            # Test if we can import our data modules
            from data.synthetic_generator import EEGGenerator, BrainState
            from data.image_generator import ImageGenerator
            from data.dataset import BrainWaveDataset
            
            print("  ✅ Data generation modules imported")
            
            # Test EEG generation
            generator = EEGGenerator(sample_rate=250, duration=4.0)
            eeg_sample = generator.generate_brain_state_eeg(BrainState.RELAXED)
            
            if eeg_sample.shape[0] == 1000:  # 4 seconds * 250 Hz
                print("  ✅ EEG generation working")
            else:
                print(f"  ❌ EEG generation shape mismatch: {eeg_sample.shape}")
                return False
            
            # Test image generation
            img_generator = ImageGenerator(image_size=(64, 64))
            img_sample = img_generator.generate_brain_state_image(0, variation=0.5)
            
            if img_sample.shape == (64, 64, 3):
                print("  ✅ Image generation working")
            else:
                print(f"  ❌ Image generation shape mismatch: {img_sample.shape}")
                return False
            
            # Test dataset creation
            dataset = BrainWaveDataset(data_dir='test_data')
            test_dataset = dataset.generate_dataset(n_samples=10, save=False)
            
            if 'train_eeg' in test_dataset and 'train_images' in test_dataset:
                print("  ✅ Dataset creation working")
            else:
                print("  ❌ Dataset creation failed")
                return False
            
            return True
        except Exception as e:
            print(f"  ❌ Data generation error: {e}")
            return False
    
    def test_model_architecture(self):
        """Test model architectures"""
        try:
            # Test if we can import model modules
            from models.eeg_to_image import create_eeg_to_image_model
            from models.image_to_eeg import create_image_to_eeg_model
            from models.attention import TemporalAttention, SpatialAttention
            
            print("  ✅ Model architecture modules imported")
            
            # Test EEG to Image model creation
            eeg_to_image_model = create_eeg_to_image_model(
                time_steps=100,
                n_features=1,
                image_size=(64, 64),
                latent_dim=128,  # Smaller for testing
                use_attention=True,
                use_vae=False
            )
            
            if eeg_to_image_model is not None:
                print("  ✅ EEG to Image model created")
            else:
                print("  ❌ EEG to Image model creation failed")
                return False
            
            # Test Image to EEG model creation
            image_to_eeg_model = create_image_to_eeg_model(
                image_size=(64, 64),
                time_steps=100,
                n_features=1,
                latent_dim=128,  # Smaller for testing
                use_attention=True,
                use_teacher_forcing=False
            )
            
            if image_to_eeg_model is not None:
                print("  ✅ Image to EEG model created")
            else:
                print("  ❌ Image to EEG model creation failed")
                return False
            
            # Test attention mechanisms
            temporal_attn = TemporalAttention(units=32)
            spatial_attn = SpatialAttention()
            
            print("  ✅ Attention mechanisms created")
            
            return True
        except Exception as e:
            print(f"  ❌ Model architecture error: {e}")
            return False
    
    def test_utils(self):
        """Test utility functions"""
        try:
            # Test signal processing
            from utils.signal_processing import EEGProcessor, ImageProcessor
            
            print("  ✅ Signal processing modules imported")
            
            # Test EEG processing
            eeg_processor = EEGProcessor(sample_rate=250, duration=4.0)
            test_eeg = np.random.randn(1000)
            
            processed_eeg = eeg_processor.preprocess_eeg(test_eeg)
            if processed_eeg.shape == test_eeg.shape:
                print("  ✅ EEG processing working")
            else:
                print("  ❌ EEG processing failed")
                return False
            
            # Test image processing
            img_processor = ImageProcessor(target_size=(64, 64))
            test_image = np.random.rand(100, 100, 3)
            
            processed_img = img_processor.preprocess_image(test_image)
            if processed_img.shape == (64, 64, 3):
                print("  ✅ Image processing working")
            else:
                print("  ❌ Image processing failed")
                return False
            
            # Test visualization
            from utils.visualization import EEGVisualizer, ImageVisualizer
            
            print("  ✅ Visualization modules imported")
            
            eeg_viz = EEGVisualizer()
            img_viz = ImageVisualizer()
            
            # Test basic visualization creation
            fig = eeg_viz.create_time_series_plot(test_eeg)
            if fig is not None:
                print("  ✅ EEG visualization working")
            else:
                print("  ❌ EEG visualization failed")
                return False
            
            return True
        except Exception as e:
            print(f"  ❌ Utils error: {e}")
            return False
    
    def test_evaluation(self):
        """Test evaluation metrics"""
        try:
            from evaluation.metrics import EEGMetrics, ImageMetrics, ModelEvaluator
            
            print("  ✅ Evaluation modules imported")
            
            # Test EEG metrics
            eeg_metrics = EEGMetrics()
            test_eeg1 = np.random.randn(100)
            test_eeg2 = np.random.randn(100)
            
            mse = eeg_metrics.mse(test_eeg1, test_eeg2)
            if isinstance(mse, float):
                print("  ✅ EEG metrics working")
            else:
                print("  ❌ EEG metrics failed")
                return False
            
            # Test image metrics
            img_metrics = ImageMetrics()
            test_img1 = np.random.rand(64, 64, 3)
            test_img2 = np.random.rand(64, 64, 3)
            
            img_mse = img_metrics.mse(test_img1, test_img2)
            if isinstance(img_mse, float):
                print("  ✅ Image metrics working")
            else:
                print("  ❌ Image metrics failed")
                return False
            
            return True
        except Exception as e:
            print(f"  ❌ Evaluation error: {e}")
            return False
    
    def test_api_structure(self):
        """Test API structure"""
        try:
            # Test if API file exists and has correct structure
            if not Path('api.py').exists():
                print("  ❌ API file missing")
                return False
            
            # Read API file and check for key components
            with open('api.py', 'r') as f:
                api_content = f.read()
            
            required_components = [
                'FastAPI',
                'eeg_to_image',
                'image_to_eeg',
                'health',
                'models'
            ]
            
            for component in required_components:
                if component in api_content:
                    print(f"  ✅ API contains {component}")
                else:
                    print(f"  ❌ API missing {component}")
                    return False
            
            return True
        except Exception as e:
            print(f"  ❌ API structure error: {e}")
            return False
    
    def test_streamlit_structure(self):
        """Test Streamlit app structure"""
        try:
            # Test if Streamlit app file exists
            if not Path('app.py').exists():
                print("  ❌ Streamlit app file missing")
                return False
            
            # Read app file and check for key components
            with open('app.py', 'r') as f:
                app_content = f.read()
            
            required_components = [
                'streamlit',
                'st.set_page_config',
                'EEG → Image',
                'Image → EEG',
                'Analysis',
                'About'
            ]
            
            for component in required_components:
                if component in app_content:
                    print(f"  ✅ Streamlit app contains {component}")
                else:
                    print(f"  ❌ Streamlit app missing {component}")
                    return False
            
            return True
        except Exception as e:
            print(f"  ❌ Streamlit structure error: {e}")
            return False
    
    def test_demo_launcher(self):
        """Test demo launcher"""
        try:
            # Test if demo launcher exists
            if not Path('run_demo.py').exists():
                print("  ❌ Demo launcher missing")
                return False
            
            # Read launcher and check for key components
            with open('run_demo.py', 'r') as f:
                launcher_content = f.read()
            
            required_components = [
                'BrainWaveDemoLauncher',
                'check_dependencies',
                'start_api_server',
                'start_streamlit_app',
                'run_demo'
            ]
            
            for component in required_components:
                if component in launcher_content:
                    print(f"  ✅ Demo launcher contains {component}")
                else:
                    print(f"  ❌ Demo launcher missing {component}")
                    return False
            
            return True
        except Exception as e:
            print(f"  ❌ Demo launcher error: {e}")
            return False
    
    def test_configuration(self):
        """Test configuration system"""
        try:
            from training.config import Config, get_demo_config, get_full_config
            
            print("  ✅ Configuration modules imported")
            
            # Test demo config
            demo_config = get_demo_config()
            if demo_config.training.n_samples > 0:
                print("  ✅ Demo configuration working")
            else:
                print("  ❌ Demo configuration failed")
                return False
            
            # Test full config
            full_config = get_full_config()
            if full_config.training.n_samples > demo_config.training.n_samples:
                print("  ✅ Full configuration working")
            else:
                print("  ❌ Full configuration failed")
                return False
            
            return True
        except Exception as e:
            print(f"  ❌ Configuration error: {e}")
            return False
    
    def test_integration(self):
        """Test integration between components"""
        try:
            # Test if models.py can be imported and used
            from models import DemoModels
            
            print("  ✅ Models wrapper imported")
            
            # Test model initialization
            models = DemoModels()
            
            if hasattr(models, 'predict_image_from_eeg') and hasattr(models, 'predict_eeg_from_image'):
                print("  ✅ Model wrapper has required methods")
            else:
                print("  ❌ Model wrapper missing required methods")
                return False
            
            # Test basic prediction (with dummy data)
            dummy_eeg = np.random.randn(100)
            try:
                generated_image = models.predict_image_from_eeg(dummy_eeg)
                if generated_image.shape == (64, 64, 3):
                    print("  ✅ EEG to Image prediction working")
                else:
                    print(f"  ❌ EEG to Image prediction shape mismatch: {generated_image.shape}")
                    return False
            except Exception as e:
                print(f"  ❌ EEG to Image prediction failed: {e}")
                return False
            
            dummy_image = np.random.rand(64, 64, 3)
            try:
                generated_eeg = models.predict_eeg_from_image(dummy_image)
                if len(generated_eeg) == 100:
                    print("  ✅ Image to EEG prediction working")
                else:
                    print(f"  ❌ Image to EEG prediction length mismatch: {len(generated_eeg)}")
                    return False
            except Exception as e:
                print(f"  ❌ Image to EEG prediction failed: {e}")
                return False
            
            return True
        except Exception as e:
            print(f"  ❌ Integration error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all component tests"""
        print("🧠 BrainWave Analyzer - Component Testing Suite")
        print("=" * 60)
        
        # Run all tests
        self.run_test("Basic Imports", self.test_imports)
        self.run_test("Data Generation", self.test_data_generation)
        self.run_test("Model Architecture", self.test_model_architecture)
        self.run_test("Utility Functions", self.test_utils)
        self.run_test("Evaluation Metrics", self.test_evaluation)
        self.run_test("API Structure", self.test_api_structure)
        self.run_test("Streamlit Structure", self.test_streamlit_structure)
        self.run_test("Demo Launcher", self.test_demo_launcher)
        self.run_test("Configuration System", self.test_configuration)
        self.run_test("Component Integration", self.test_integration)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        print(f"  Total Tests: {self.total_tests}")
        print(f"  Passed: {self.passed_tests}")
        print(f"  Failed: {self.total_tests - self.passed_tests}")
        print(f"  Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        for test_name, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {status}: {test_name}")
        
        # Overall status
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! Demo is ready! 🚀")
            return True
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} TESTS FAILED. Please review and fix issues.")
            return False


def main():
    """Main testing function"""
    tester = ComponentTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Next Steps:")
        print("  1. Run: python run_demo.py")
        print("  2. Open browser to Streamlit interface")
        print("  3. Demo is ready!")
    else:
        print("\n🔧 Troubleshooting:")
        print("  1. Check error messages above")
        print("  2. Install missing dependencies: pip install -r requirements.txt")
        print("  3. Verify all files are present")
        print("  4. Re-run tests after fixes")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
