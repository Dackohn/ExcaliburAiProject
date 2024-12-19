import React from "react";

const SentimentResults = ({ data }) => {
  return (
    <div className="sentiment-results">
      <h2>Sentiment Results</h2>
      <p><strong>Overall Sentiment:</strong> {data.sentiment}</p>
      <p><strong>Confidence Score:</strong> {data.score.toFixed(2)}</p>
    </div>
  );
};

export default SentimentResults;
