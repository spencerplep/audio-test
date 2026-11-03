"""Live microphone spectrogram. Install: py -m pip install -r requirements.txt"""
import sys
from collections import deque

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import sounddevice as sd

FFT_SIZE = 2048
HOP = 512
HISTORY_SECONDS = 10
DB_MIN, DB_MAX = -100, 0


class Spectrogram(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live audio spectrogram")
        self.resize(1000, 600)
        self.stream = None
        self.blocks = deque(maxlen=256)
        self.pending = np.empty(0, dtype=np.float32)
        self.window = np.hanning(FFT_SIZE)

        self.inputs = QtWidgets.QComboBox()
        self.button = QtWidgets.QPushButton("Start")
        self.button.clicked.connect(self.toggle)
        self.status = QtWidgets.QLabel("Select an input, then click Start. Uses channel 1.")
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.inputs, 1)
        controls.addWidget(self.button)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time relative to now", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.plot.addItem(self.image)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)

        try:
            apis = sd.query_hostapis()
            for index, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    api = apis[device["hostapi"]]["name"]
                    self.inputs.addItem(f"{device['name']} ({api})", index)
            default = self.inputs.findData(sd.default.device[0])
            if default >= 0:
                self.inputs.setCurrentIndex(default)
            if not self.inputs.count():
                self.status.setText("No audio inputs found.")
                self.button.setEnabled(False)
        except sd.PortAudioError as exc:
            self.status.setText(str(exc))
            self.button.setEnabled(False)

    def toggle(self):
        if self.stream is not None:
            self.stop()
            return
        try:
            device = self.inputs.currentData()
            rate = sd.query_devices(device)["default_samplerate"]
            self.blocks.clear()
            self.pending = np.empty(0, dtype=np.float32)
            columns = max(1, round(HISTORY_SECONDS * rate / HOP))
            self.history = np.full((FFT_SIZE // 2 + 1, columns), DB_MIN,
                                   dtype=np.float32)
            self.image.setImage(self.history, levels=(DB_MIN, DB_MAX))
            duration = columns * HOP / rate
            bin_width = rate / FFT_SIZE
            self.image.setRect(QtCore.QRectF(-duration, -bin_width / 2,
                                            duration, rate / 2 + bin_width))
            self.plot.setRange(xRange=(-duration, 0), yRange=(0, rate / 2), padding=0)
            self.stream = sd.InputStream(device=device, samplerate=rate,
                                        channels=1, dtype="float32", blocksize=HOP,
                                        callback=self.audio_callback)
            self.stream.start()
            self.inputs.setEnabled(False)
            self.button.setText("Stop")
            self.status.setText(f"{rate:g} Hz • Channel 1 • Color: {DB_MIN} to {DB_MAX} dBFS")
            self.timer.start(30)
        except Exception as exc:
            self.stop()
            self.status.setText(f"Could not open input: {exc}")

    def audio_callback(self, data, frames, timing, status):
        # No Qt calls in the audio thread; bound memory if the UI falls behind.
        self.blocks.append((data[:, 0].copy(), str(status)))

    def update_plot(self):
        blocks = []
        for _ in range(len(self.blocks)):
            data, warning = self.blocks.popleft()
            blocks.append(data)
            if warning:
                self.status.setText(f"Audio warning: {warning}")
        if not blocks:
            return
        self.pending = np.concatenate([self.pending, *blocks])
        spectra = []
        while self.pending.size >= FFT_SIZE:
            amplitude = np.abs(np.fft.rfft(self.pending[:FFT_SIZE] * self.window))
            amplitude *= 2 / self.window.sum()
            amplitude[[0, -1]] *= 0.5
            spectra.append(20 * np.log10(np.maximum(amplitude, 1e-10)))
            self.pending = self.pending[HOP:]
        if spectra:
            new = np.asarray(spectra).T[:, -self.history.shape[1]:]
            count = new.shape[1]
            self.history = np.roll(self.history, -count, axis=1)
            self.history[:, -count:] = new
            self.image.setImage(self.history, autoLevels=False)

    def stop(self):
        self.timer.stop()
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.inputs.setEnabled(True)
        self.button.setText("Start")
        self.status.setText("Stopped.")

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = Spectrogram()
    viewer.show()
    sys.exit(app.exec())
