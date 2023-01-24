using Microsoft.ML.Data;


namespace ImageRecognitionOnnxSample.Data
{
    public class ImageData
    {
        [LoadColumn(0), ColumnName("ImagePath")]
        public string ImagePath { get; set; }
    }
}
