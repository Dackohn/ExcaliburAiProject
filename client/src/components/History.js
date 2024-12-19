import React from "react";

const History = ({ history }) => {
  return (
    <div className="history">
      <h2>Analysis History</h2>
      {history.map((item, index) => (
        <div key={index} className="history-item">
          <p>
            <strong>Time:</strong> {item.time}
          </p>
          <p>
            <strong>Input Text:</strong> {item.text}
          </p>
          <p>
            <strong>Sentiment:</strong> {item.sentiment} ({item.score})
          </p>
        </div>
      ))}
    </div>
  );
};

export default History;
