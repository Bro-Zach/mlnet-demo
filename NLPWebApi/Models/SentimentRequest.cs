using System.ComponentModel.DataAnnotations;

namespace NLPWebApi.Models
{
    public class SentimentRequest
    {
        [Required]
        public string SentimentText { get; set; }
    }
}
