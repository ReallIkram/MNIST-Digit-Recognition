import { useRef } from "react";
import SignatureCanvas from "react-signature-canvas";

function DrawingCanvas({ onPredict }) {
    const canvasRef = useRef(null);

    // ========================================================
    // CLEAR CANVAS
    // ========================================================

    const clearCanvas = () => {
        canvasRef.current.clear();
    };


    // ========================================================
    // PREDICT DIGIT
    // ========================================================

    const predictDigit = () => {

        // Check whether canvas exists
        if (!canvasRef.current) {
            return;
        }

        // Check whether user actually drew something
        if (canvasRef.current.isEmpty()) {

            alert("Please draw a digit first.");

            return;
        }

        // Convert canvas drawing to PNG
        const imageData =
            canvasRef.current.toDataURL(
                "image/png"
            );

        // Send image to parent component
        onPredict(imageData);
    };


    // ========================================================
    // RENDER
    // ========================================================

    return (
        <div className="drawing-container">

            <div className="canvas-wrapper">

                <SignatureCanvas

                    ref={canvasRef}

                    penColor="white"

                    backgroundColor="black"

                    canvasProps={{
                        width: 400,
                        height: 400,
                        className: "drawing-canvas"
                    }}

                    minWidth={8}

                    maxWidth={12}
                />

            </div>


            <div className="canvas-buttons">

                <button
                    onClick={clearCanvas}
                >
                    Clear
                </button>


                <button
                    onClick={predictDigit}
                >
                    Predict
                </button>

            </div>

        </div>
    );
}

export default DrawingCanvas;