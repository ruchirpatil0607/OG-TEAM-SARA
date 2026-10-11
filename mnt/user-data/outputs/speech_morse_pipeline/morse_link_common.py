"""
morse_link_common.py — shared code for the Acoustic Morse Link (Project 01).

Text <-> Morse <-> audio tone bursts, plus an FFT-based tone/silence
decoder. No hardware needed beyond a speaker and a mic.
"""

import numpy as np
from scipy.signal import find_peaks

FS = 44100
TONE_FREQ = 900          # Hz — a single tone for Morse (on/off keying)
UNIT = 0.08               # seconds — length of one Morse "dot"

MORSE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    ' ': '/',
}
MORSE_INV = {v: k for k, v in MORSE.items()}


def text_to_morse(text):
    return ' '.join(MORSE.get(ch.upper(), '') for ch in text if ch.upper() in MORSE)


def morse_to_text(morse):
    words = morse.strip().split(' / ')
    out = []
    for word in words:
        letters = word.split(' ')
        out.append(''.join(MORSE_INV.get(l, '') for l in letters if l))
    return ' '.join(out)


# ---------------------------------------------------------------------
# Morse -> audio
# ---------------------------------------------------------------------

def _tone(duration, freq=TONE_FREQ, fs=FS):
    t = np.arange(0, duration, 1 / fs)
    return np.sin(2 * np.pi * freq * t)


def _silence(duration, fs=FS):
    return np.zeros(int(duration * fs))


def morse_to_audio(morse, unit=UNIT, fs=FS):
    """Render a Morse string ('.-. / --.') to a mono audio waveform."""
    chunks = []
    for symbol in morse:
        if symbol == '.':
            chunks.append(_tone(unit, fs=fs))
            chunks.append(_silence(unit, fs=fs))          # intra-char gap
        elif symbol == '-':
            chunks.append(_tone(3 * unit, fs=fs))
            chunks.append(_silence(unit, fs=fs))
        elif symbol == ' ':
            chunks.append(_silence(2 * unit, fs=fs))       # inter-letter gap (total 3*unit incl. above)
        elif symbol == '/':
            chunks.append(_silence(6 * unit, fs=fs))       # inter-word gap
    return np.concatenate(chunks) if chunks else np.array([])


def text_to_audio(text, unit=UNIT, fs=FS):
    return morse_to_audio(text_to_morse(text), unit, fs)


# ---------------------------------------------------------------------
# audio -> Morse (energy-based tone/silence decoder)
# ---------------------------------------------------------------------

def _envelope(signal, fs, smooth_ms=10):
    """Rectify + smooth to get an amplitude envelope."""
    rectified = np.abs(signal)
    win = max(1, int(fs * smooth_ms / 1000))
    kernel = np.ones(win) / win
    return np.convolve(rectified, kernel, mode='same')


def audio_to_morse(signal, unit=UNIT, fs=FS, threshold_ratio=0.3):
    """
    Decode a recorded waveform back to a Morse string by scanning the
    envelope for tone-on / tone-off segments and classifying each
    tone-on run as a dot or a dash by its duration, and each gap as an
    intra-char / inter-letter / inter-word silence.
    """
    env = _envelope(signal, fs)
    threshold = env.max() * threshold_ratio
    is_on = env > threshold

    # find runs of on/off
    runs = []
    current = is_on[0]
    start = 0
    for i in range(1, len(is_on)):
        if is_on[i] != current:
            runs.append((current, (i - start) / fs))
            start = i
            current = is_on[i]
    runs.append((current, (len(is_on) - start) / fs))

    symbols = []
    for is_tone, duration in runs:
        n_units = duration / unit
        if is_tone:
            symbols.append('.' if n_units < 2 else '-')
        else:
            if n_units < 1.5:
                continue                       # intra-character gap, no symbol
            elif n_units < 4.5:
                symbols.append(' ')             # inter-letter gap
            else:
                symbols.append(' / ')           # inter-word gap

    morse = ''.join(symbols)
    # collapse accidental double spaces from gap classification
    morse = ' '.join(morse.split())
    return morse


if __name__ == "__main__":
    msg = "SOS HELLO"
    audio = text_to_audio(msg)
    decoded_morse = audio_to_morse(audio)
    decoded_text = morse_to_text(decoded_morse)
    print("sent:   ", msg)
    print("morse:  ", text_to_morse(msg))
    print("decoded morse:", decoded_morse)
    print("decoded text: ", decoded_text)
    print("match:", decoded_text == msg)
