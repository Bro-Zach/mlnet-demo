using System;
using System.IO;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using static System.Console;
using Microsoft.ML.Data;
using System.Collections.Generic;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static void Main(string[] args)
        {
            MLContext context = new MLContext();
            TrainTestData splitDataView = LoadData(context);
            ITransformer model = BuildAndTrainModel(context, splitDataView.TrainSet);
            Evaluate(context, model, splitDataView.TestSet);
            UseModelWithSingleItem(context, model);
            UseModelWithBatchItems(context, model);
            ReadLine();
        }

        private static void UseModelWithBatchItems(MLContext context, ITransformer model)
        {
            var sampleData = new[] {
                new SentimentData
                {
                    SentimentText="This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText="I love this spaghetti."
                }
            };

            IDataView batchComments = context.Data.LoadFromEnumerable(sampleData);
            IDataView predictions = model.Transform(batchComments);
            IEnumerable<SentimentPrediction> predictionResult = context.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            WriteLine();

            WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPrediction prediction in predictionResult)
            {
                WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            WriteLine("=============== End of predictions ===============");
        }

        /// <summary>
        /// Creates a single comment of test data.
        /// Predicts sentiment based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="model">The model.</param>
        private static void UseModelWithSingleItem(MLContext context, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            var sampleData = new SentimentData
            {
                SentimentText = "this was an extremely bad steak"
            };
            var predictionResult = predictionEngine.Predict(sampleData);
            WriteLine();
            WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            WriteLine();
            WriteLine($"Sentiment: {predictionResult.SentimentText} | Prediction: {(Convert.ToBoolean(predictionResult.Prediction) ? "Positive" : "Negative")} | Probability: {predictionResult.Probability} ");

            WriteLine("=============== End of Predictions ===============");
            WriteLine();
        }


        /// <summary>
        /// Loads the test dataset.
        /// Creates the BinaryClassification evaluator.
        /// Evaluates the model and creates metrics.
        /// Displays the metrics.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="model">The model.</param>
        /// <param name="testSet">The test set.</param>
        private static void Evaluate(MLContext context, ITransformer model, IDataView testSet)
        {
            WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(testSet);
            CalibratedBinaryClassificationMetrics metrics = context.BinaryClassification.Evaluate(predictions, "Label");
            WriteLine();
            WriteLine("Model quality metrics evaluation");
            WriteLine("--------------------------------");
            WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            WriteLine($"F1Score: {metrics.F1Score:P2}");
            WriteLine("=============== End of model evaluation ===============");
        }

        /// <summary>
        /// Builds the and train model.
        /// Extracts and transforms the data.
        /// Trains the model.
        /// Predicts sentiment based on test data.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="trainSet">The train set.</param>
        /// <returns>Returns the model.</returns>
        private static ITransformer BuildAndTrainModel(MLContext context, IDataView trainSet)
        {
            var estimator = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(trainSet);
            WriteLine("=============== End of training ===============");
            WriteLine();

            //context.Model.Save(model, trainSet.Schema, "sentimentanalysismodel.zip");

            return model;
        }

        /// <summary>
        /// Loads the data.
        /// Splits the loaded dataset into train and test datasets.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <returns>Returns the split train and test datasets.</returns>
        private static TrainTestData LoadData(MLContext context)
        {
            IDataView dataView = context.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = context.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
    }
}
