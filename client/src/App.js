import React, { useState } from "react";
import "./styles.css";
import InputArea from "./components/InputArea";
import History from "./components/History";

function App() {
  const [showIntro, setShowIntro] = useState(true);
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

  if (showIntro) {
    return (
      <div className="intro-screen">
        <div className="intro-content">
          <h1>Sentiment Analysis Tool</h1>
          <p>
            Discover the emotional tone behind any piece of text with our AI-powered sentiment analysis tool.
          </p>
          <button onClick={() => setShowIntro(false)}>Enter Application</button>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header>
        <h1>Sentiment Analysis Tool</h1>
      </header>
      <InputArea onAnalyze={(data) => handleSentimentAnalysis(data, data.text)} />
      {sentimentData && (
        <div className="sentiment-results">
          <h2>Sentiment Results</h2>
          <p>
            <strong>Sentiment:</strong> {sentimentData.sentiment}
          </p>
          <p>
            <strong>Score:</strong> {sentimentData.score}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
