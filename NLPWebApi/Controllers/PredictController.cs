using System;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using NLPWebApi.DataModels;
using NLPWebApi.Models;

namespace NLPWebApi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PredictController : ControllerBase
    {
        private readonly PredictionEnginePool<SentimentData, SentimentPrediction> _predictionEnginePool;

        public PredictController(PredictionEnginePool<SentimentData, SentimentPrediction> predictionEnginePool) =>
            _predictionEnginePool = predictionEnginePool;

        [HttpPost]
        public ActionResult<string> Post(SentimentRequest request)
        {
            SentimentData input = new SentimentData
            {
                SentimentText = request.SentimentText
            };
            SentimentPrediction prediction = _predictionEnginePool.Predict("SentimentAnalysisModel", input);
            string sentiment = Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative";
            return Ok(sentiment);
        }
    }
}
