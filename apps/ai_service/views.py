# apps/ai_service/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
from .services import detect_from_image

@csrf_exempt
def ai_detect(request):
    """
    Accepts POST with a webcam snapshot and returns YOLO detections
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        file = request.FILES.get("file")
        if not file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        # Convert uploaded file to OpenCV image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return JsonResponse({"error": "Invalid image uploaded"}, status=400)

        result = detect_from_image(frame)
        return JsonResponse(result)

    except Exception as e:
        import traceback
        return JsonResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status=500
        )
