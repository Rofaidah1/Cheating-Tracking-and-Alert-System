try:
    from plyer import notification
except ImportError:
    notification = None


def send_cheating_notification(behavior, confidence):
    title = "Cheating Alert Detected"
    message = (
        f"Behavior: {behavior}\n"
        f"Confidence: {confidence:.1%}\n"
        "Screenshot saved successfully."
    )

    if notification is not None:
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="Cheating Tracking System",
                timeout=5,
            )
            return
        except Exception:
            pass

    print("\n" + "=" * 50)
    print(title)
    print(message)
    print("=" * 50 + "\n")
