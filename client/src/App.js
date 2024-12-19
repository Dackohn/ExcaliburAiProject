import React, { useState } from "react";
import "./styles.css";
import InputArea from "./components/InputArea";
import History from "./components/History";

function App() {
  const [sentimentData, setSentimentData] = useState(null);
  const [history, setHistory] = useState([]);

  const handleSentimentAnalysis = (data, inputText) => {
    const timestamp = new Date().toLocaleString();
    const resultWithTime = { ...data, time: timestamp, text: inputText };
    setSentimentData(data);
    setHistory((prevHistory) => {
      const newHistory = [...prevHistory, resultWithTime];
      if (newHistory.length > 10) newHistory.shift();
      return newHistory;
    });
  };

  return (
    <div className="App">
      <header>
        <h1>Sentiment Analysis Tool</h1>
      </header>
      <InputArea onAnalyze={(data) => handleSentimentAnalysis(data, data.text)} />
      {sentimentData && (
        <div className="box">
        <div className="sentiment-results">
          <h2>Sentiment Results</h2>
          <p>
            <strong>Sentiment:</strong> {sentimentData.sentiment}
          </p>
          <p>
            <strong>Score:</strong> {sentimentData.score}
          </p>
        </div>
        </div>
      )}
      <div className="box">
      <History history={history} />
      </div>
    </div>
  );
}

export default App;
