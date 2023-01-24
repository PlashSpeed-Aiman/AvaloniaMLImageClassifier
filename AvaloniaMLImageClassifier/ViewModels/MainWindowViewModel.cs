using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Avalonia.Controls;
using Avalonia.Media.Imaging;
using AvaloniaMLImageClassifier.Classifier;
using ImageRecognitionOnnxSample.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using ReactiveUI;

namespace AvaloniaMLImageClassifier.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
    {
        private PredictionEngine<ImageData, ImagePrediction>? _imageOnnxClassifier = null;
        private MobileNetOnnxClassification? _mobileNetOnnx = null;
        string _onnxModelPath = String.Empty;
        string _greeting = "Welcome to Avalonia";
        // string _imagePath = Path.Combine(Directory.GetCurrentDirectory(), "images/download.jpg");
        string? _imagePath = String.Empty;
        private Avalonia.Media.Imaging.Bitmap chessboard = null;
        
        List<string> labels = File.ReadLines(Path.Combine(Directory.GetCurrentDirectory(), "labels.txt")).ToList();
        
        public Avalonia.Media.Imaging.Bitmap DisplayImagePath
        {
            get => chessboard;
            set
            {
                chessboard = value;
                this.RaisePropertyChanged("DisplayImagePath");
            }

        }

        public void LoadImage()
        {
            var bitmap = new Bitmap(_imagePath);
            chessboard = bitmap;
        }
        public string Greeting {
            get=> _greeting;
            set
            {
                _greeting = value;
                this.RaisePropertyChanged("Greeting");
            }
            
    }

        public string PathName
        {
            get => _imagePath;
            set
            {
                _imagePath = value;
                LoadImage();
                this.RaisePropertyChanged("PathName");
                this.RaisePropertyChanged("DisplayImagePath");

            }
        }

        public void ClassifyImage()
        {
            var predictionOnnx = _imageOnnxClassifier.Predict(new ImageData { ImagePath = _imagePath });
            Greeting = $"Index:{predictionOnnx.Index}\nLabel:{predictionOnnx.Label}\nEstimate:{predictionOnnx.Estimate}" ;
        }

        public MainWindowViewModel()
        {
            _onnxModelPath = Path.Combine(Directory.GetCurrentDirectory(), "mobilenetv2-1.0.onnx");
            _mobileNetOnnx = new MobileNetOnnxClassification(new Microsoft.ML.MLContext(), _onnxModelPath, labels);
            _imageOnnxClassifier = _mobileNetOnnx.CreateClassifier();
            // var predictionOnnx = _imageOnnxClassifier.Predict(new ImageData { ImagePath = _imagePath });
            // LoadImage();
            // Greeting = $"Index:{predictionOnnx.Index}\nLabel:{predictionOnnx.Label}\nEstimate:{predictionOnnx.Estimate}" ;
        }
    }
}