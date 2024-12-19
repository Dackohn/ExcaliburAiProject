import React, { useState, useEffect } from "react";
import "./style.css";

const InputArea = ({ onAnalyze }) => {
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [typingTimeout, setTypingTimeout] = useState(null);

  const analyzeText = async (inputText) => {
    if (!inputText.trim()) {
      setError("Please enter text for analysis.");
      return;
    }
    setError("");
    try {
      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) throw new Error("Error analyzing text. Try again.");
      const result = await response.json();
      onAnalyze({ ...result, text: inputText });
    } catch (err) {
      setError(err.message);
    }
  };

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);

    if (typingTimeout) clearTimeout(typingTimeout);

    const timeout = setTimeout(() => {
      analyzeText(newText);
    }, 2000); 

    setTypingTimeout(timeout);
  };

  const handleButtonClick = () => {
    analyzeText(text);
  };

  return (
    <div className="input-area">
      <textarea
        placeholder="Enter text for sentiment analysis..."
        value={text}
        onChange={handleTextChange}
      />
      <button onClick={handleButtonClick}>Analyze Sentiment</button>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default InputArea;
