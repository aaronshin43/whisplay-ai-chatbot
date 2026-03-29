from PyQt5.QtCore import QObject, pyqtSignal


class GPIOHandler(QObject):
    button_pressed  = pyqtSignal()
    button_released = pyqtSignal()

    BUTTON_PIN = 11  # Whisplay HAT physical pin

    def start(self):
        import RPi.GPIO as GPIO
        self._GPIO = GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(
            self.BUTTON_PIN, GPIO.BOTH,
            callback=self._on_edge, bouncetime=50,
        )

    def _on_edge(self, channel):
        if self._GPIO.input(channel) == self._GPIO.HIGH:
            self.button_pressed.emit()
        else:
            self.button_released.emit()

    def cleanup(self):
        self._GPIO.cleanup()
