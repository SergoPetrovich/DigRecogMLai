using System.Collections.Generic;
using System.Windows;
using System.Windows.Input;
using System.Windows.Shapes;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;


namespace DigitRecognitionApp
{
    public partial class MainWindow : Window
    {
        private bool isDrawing;
        private object model;
        private readonly List<Point> points;
        private readonly MLContext mlContext;
        private readonly PredictionEngine<DigitDrawing, DigitPrediction> predictionEngine;

        public MainWindow()
        {
            InitializeComponent();

            isDrawing = false;
            points = new List<Point>();

            mlContext = new MLContext();
            //var onnx_model = mlContext.Transforms.LoadOnnxModel("model.onnx");
            predictionEngine = mlContext.Model.CreatePredictionEngine<DigitDrawing, DigitPrediction>(LoadModel());
        }

        private ITransformer LoadModel()
        {
            var modelFilePath = "model.onnx";
            return mlContext.Transforms.LoadOnnxModel(modelFilePath, out _);
        }

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (isDrawing)
            {
                points.Add(e.GetPosition((IInputElement)sender));
                DrawPoints();
            }
        }

        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            isDrawing = true;
            points.Clear();
            points.Add(e.GetPosition((IInputElement)sender));
        }

        private void Canvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            isDrawing = false;
            RecognizeDigit();
        }

        private void DrawPoints()
        {
            canvas.Children.Clear();

            for (int i = 0; i < points.Count - 1; i++)
            {
                var line = new Line
                {
                    X1 = points[i].X,
                    Y1 = points[i].Y,
                    X2 = points[i + 1].X,
                    Y2 = points[i + 1].Y,
                    Stroke = System.Windows.Media.Brushes.Black,
                    StrokeThickness = 3
                };
                canvas.Children.Add(line);
            }
        }


        private void RecognizeDigit()
        {
            var digitDrawing = new DigitDrawing
            {
                Pixels = GetPixelsFromPoints(points)
            };

            var prediction = predictionEngine.Predict(digitDrawing);
            var recognizedDigit = prediction.PredictedLabel;

            MessageBox.Show($"Recognized digit: {recognizedDigit}");
        }

        private float[] GetPixelsFromPoints(List<Point> points)
        {
            const int imageSize = 28;
            var pixels = new float[imageSize * imageSize];

            foreach (var point in points)
            {
                var x = (int)(point.X / (canvas.ActualWidth / imageSize));
                var y = (int)(point.Y / (canvas.ActualHeight / imageSize));
                var index = y * imageSize + x;
                pixels[index] = 1;
            }

            return pixels;
        }

        private void RecognizeButton_Click(object sender, RoutedEventArgs e)
        {
            
                var digitDrawing = new DigitDrawing
                {
                    Pixels = GetPixelsFromPoints(points)
                };

                var prediction = predictionEngine.Predict(digitDrawing);
                var recognizedDigit = prediction.PredictedLabel;

                MessageBox.Show($"Recognized digit: {recognizedDigit}");
            

        }
    }

    public class DigitDrawing
    {
        [VectorType(784)]
        public float[] Pixels { get; set; }
    }

    public class DigitPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedLabel;
    }
}


