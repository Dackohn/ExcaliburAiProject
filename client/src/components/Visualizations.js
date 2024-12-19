import React from "react";
import { Pie } from "react-chartjs-2";
import "chart.js/auto";

const Visualizations = ({ data }) => {
  const chartData = {
    labels: ["Positive", "Negative", "Neutral"],
    datasets: [
      {
        data: [data.positive, data.negative, data.neutral],
        backgroundColor: ["#4caf50", "#f44336", "#9e9e9e"],
      },
    ],
  };

  return (
    <div className="visualizations">
      <h2>Sentiment Distribution</h2>
      <Pie data={chartData} />
    </div>
  );
};

export default Visualizations;
