import { useState } from "react";

import DrawingCanvas from "./components/DrawingCanvas";

import "./App.css";


function App() {

    // ========================================================
    // STATE
    // ========================================================

    const [prediction, setPrediction] =
        useState(null);

    const [loading, setLoading] =
        useState(false);

    const [error, setError] =
        useState(null);


    // ========================================================
    // SEND IMAGE TO FASTAPI
    // ========================================================

    const handlePredict = async (imageData) => {

        try {

            // ------------------------------------------------
            // Reset previous state
            // ------------------------------------------------

            setLoading(true);

            setError(null);

            setPrediction(null);


            // ------------------------------------------------
            // Convert Base64 image to Blob
            // ------------------------------------------------

            const response = await fetch(
                imageData
            );

            const blob = await response.blob();


            // ------------------------------------------------
            // Create FormData
            // ------------------------------------------------

            const formData = new FormData();

            formData.append(
                "file",
                blob,
                "drawing.png"
            );


            // ------------------------------------------------
            // Send image to FastAPI
            // ------------------------------------------------

            const apiResponse = await fetch(
                "http://127.0.0.1:8000/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


            // ------------------------------------------------
            // Check HTTP response
            // ------------------------------------------------

            if (!apiResponse.ok) {

                const errorData =
                    await apiResponse.json();

                throw new Error(
                    errorData.detail ||
                    "Prediction request failed."
                );
            }


            // ------------------------------------------------
            // Convert response to JSON
            // ------------------------------------------------

            const result =
                await apiResponse.json();


            // ------------------------------------------------
            // Store prediction
            // ------------------------------------------------

            setPrediction(result);


        } catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            setError(
                error.message ||
                "Something went wrong."
            );

        } finally {

            setLoading(false);
        }
    };


    // ========================================================
    // RENDER
    // ========================================================

    return (

        <div className="app">

            <h1>
                MNIST Digit Recognition
            </h1>


            <p>
                Draw a digit from 0 to 9
            </p>


            {/* =================================================
                DRAWING CANVAS
            ================================================= */}

            <DrawingCanvas
                onPredict={handlePredict}
            />


            {/* =================================================
                LOADING
            ================================================= */}

            {loading && (

                <div className="loading">

                    <p>
                        AI is analyzing your digit...
                    </p>

                </div>

            )}


            {/* =================================================
                ERROR
            ================================================= */}

            {error && (

                <div className="error">

                    <p>
                        {error}
                    </p>

                </div>

            )}


            {/* =================================================
                PREDICTION
            ================================================= */}

            {prediction && !loading && (

                <div className="prediction">

                    <h2>
                        Prediction
                    </h2>


                    <div className="predicted-digit">

                        {prediction.digit}

                    </div>


                    <p className="confidence">

                        Confidence:{" "}

                        {(
                            prediction.confidence * 100
                        ).toFixed(2)}

                        %

                    </p>


                    {/* =========================================
                        PROBABILITIES
                    ========================================= */}

                    <div className="probabilities">

                        <h3>
                            Probability Distribution
                        </h3>


                        {Object.entries(
                            prediction.probabilities
                        ).map(
                            ([digit, probability]) => (

                                <div
                                    className="probability-row"
                                    key={digit}
                                >

                                    <span className="digit-label">
                                        {digit}
                                    </span>


                                    <div className="probability-bar">

                                        <div
                                            className="probability-fill"
                                            style={{
                                                width:
                                                    `${probability * 100}%`
                                            }}
                                        />

                                    </div>


                                    <span className="probability-value">

                                        {(
                                            probability * 100
                                        ).toFixed(2)}
                                        %

                                    </span>

                                </div>

                            )
                        )}

                    </div>

                </div>

            )}

        </div>
    );
}


export default App;