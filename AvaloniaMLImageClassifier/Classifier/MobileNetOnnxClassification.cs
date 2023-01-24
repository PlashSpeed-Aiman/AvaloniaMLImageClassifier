using ImageRecognitionOnnxSample.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Transforms.Onnx;

namespace AvaloniaMLImageClassifier.Classifier
{
    public class MobileNetOnnxClassification
    {
        public const string OutputName = "mobilenetv20_output_flatten0_reshape0";
        private const int ImageHeight = 224;
        private const int ImageWidth = 224;

        private MLContext _mlContext;
        private string _modelFilePath;
        private List<string> _labels;

        public MobileNetOnnxClassification(MLContext mlContext, string modelFilePath, List<string> labels)
        {
            _mlContext = mlContext;
            _modelFilePath = modelFilePath;
            _labels = labels;
        }

        public PredictionEngine<ImageData, ImagePrediction> CreateClassifier()
        {
            var data = _mlContext.Data.LoadFromEnumerable(new List<ImageData>());
            var pipeline = _mlContext.Transforms.LoadImages(outputColumnName: "ImageData", imageFolder: string.Empty,
                        inputColumnName: "ImagePath")
                    .Append(
                        _mlContext.Transforms.ResizeImages("ImageResized", imageWidth: ImageWidth,
                            imageHeight: ImageHeight, inputColumnName: "ImageData")
                    )
                    .Append(
                        _mlContext.Transforms.ExtractPixels(outputColumnName:"Red", inputColumnName: "ImageResized",
                            colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Red, offsetImage: 0.485f * 255,
                            scaleImage: 1 / (0.229f * 255))
                    )
                    .Append(
                        _mlContext.Transforms.ExtractPixels(outputColumnName: "Green", inputColumnName: "ImageResized",
                            colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Green, offsetImage: 0.456f * 255,
                            scaleImage: 1 / (0.224f * 255))
                    )
                    .Append(
                        _mlContext.Transforms.ExtractPixels(outputColumnName:"Blue", inputColumnName: "ImageResized",
                            colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Blue, offsetImage: 0.406f * 255,
                            scaleImage: 1 / (0.225f * 255))
                    )
                    .Append(
                        _mlContext.Transforms.Concatenate("data", "Red", "Green", "Blue")
                    )
                    
                    .Append(_mlContext.Transforms.ApplyOnnxModel(modelFile: _modelFilePath,
                        outputColumnNames: new string[] { @"mobilenetv20_output_flatten0_reshape0" },
                        inputColumnNames: new string[] { "data" }))
                    .Append(_mlContext.Transforms.CustomMapping<MobileNetOnnxPrediction,ImagePrediction>(
                        mapAction: (networkResult, prediction) =>
                        {
                            prediction.Estimate = networkResult.Output.Max();
                            prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate);
                            prediction.Label = _labels[prediction.Index];
                        },
                        contractName: "MobileNetExtractor"
                    ));
              var transformers = pipeline.Fit(data);
              


            //attention code below doesn't work correctly ¯\_(ツ)_/¯
            //var pipeline = new ImageLoadingEstimator(_mlContext, string.Empty, ("ImageData", "ImagePath"))
            //        .Append(new ImageResizingEstimator(_mlContext, "ImageResized", ImageWidth, ImageHeight, "ImageData"))
            //        .Append(new ImagePixelExtractingEstimator(_mlContext, "Red", "ImageResized", colors: ColorBits.Red, offset: 0.485f * 255, scale: 1 / (0.229f * 255)))
            //        .Append(new ImagePixelExtractingEstimator(_mlContext, "Green", "ImageResized", colors: ColorBits.Green, offset: 0.456f * 255, scale: 1 / (0.224f * 255)))
            //        .Append(new ImagePixelExtractingEstimator(_mlContext, "Blue", "ImageResized", colors: ColorBits.Blue, offset: 0.406f * 255, scale: 1 / (0.225f * 255)))
            //        .Append(new ColumnConcatenatingEstimator(_mlContext, "data", "Red", "Green", "Blue"))
            //        .Append(new OnnxScoringEstimator(_mlContext, new string[] { @"mobilenetv20_output_flatten0_reshape0" }, new string[] { "data" }, _modelFilePath))
            //        .Append(new CustomMappingEstimator<MobileNetOnnxPrediction, ImagePrediction>(_mlContext, contractName: "MobileNetExtractor",
            //          mapAction: (networkResult, prediction) =>
            //          {
            //              prediction.Estimate = networkResult.Output.Max();
            //              prediction.Index = networkResult.Output.ToList().IndexOf(prediction.Estimate);
            //              prediction.Label = _labels[prediction.Index];
            //          }));

            var transformer = pipeline.Fit(data);

            return _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(transformer);
        }
    }
}
