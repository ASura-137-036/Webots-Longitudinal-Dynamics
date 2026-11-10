"""The comparison HUD, drawn with Webots `Display` primitives.

`Display` renders into a 2D texture on any `Shape` node, so with a `Shape` + `ImageTexture` child it
becomes a real screen in the 3D scene - and, being scene geometry, it is captured by
`movieStartRecording`. Everything here is drawn with `setColor`, `fillRectangle`, `drawLine`,
`drawPolygon`, `setFont` and `drawText`; no imaging library is involved.

The canvas covers the whole frame and is cleared to *fully transparent* every frame, so only the
cards are opaque and the simulation stays visible everywhere else. That is what makes the panel a
graphics overlay rather than a screen that hides the car.

Design constraints, from how the clip will actually be watched:

* LinkedIn autoplays muted at roughly 350 px wide, so text has to be large, contrast high, and there
  must be no hairlines or small legends.
* Model and measurement are told apart by **colour and lineweight**, never by a dash pattern that
  disappears when scaled down.
* The time axis moves, the value axis does not. A scrolling x-axis reads as motion; a rescaling y-axis
  reads as the data jumping.
* The car occupies the middle of the frame, so the two plots live in the upper corners and the
  numbers in the lower left, clear of the body.
"""

from __future__ import annotations

import math

# --- palette ---------------------------------------------------------------
PANEL_BACKGROUND = 0x11151C
PANEL_BORDER = 0x39424F
GRID = 0x27303B
MEASURED = 0xF2F4F8
MODEL = 0xFFB000
MUTED = 0x9AA4B2
GROUNDED = 0x35C46A
LIFT_FRONT = 0xFF7A45
LIFT_REAR = 0xC77DFF
LIFT_ALL = 0xE4572E
THROTTLE = 0x4CC2FF
BRAKE = 0xFF5D6C
SHADOW = 0x05070A

# --- layout (pixels, origin top-left as the Display API defines it) --------
# The layout is authored in a fixed 1920x1080 design space and scaled onto the canvas by `SCALE`, so
# every coordinate and font size below stays readable. The canvas itself is 1280x720 because Webots
# re-uploads, and leaks a copy of, the whole display image on every refresh: at 1920x1080 that is
# ~45 MB a refresh, which caps the panel at 8 Hz and makes the traces visibly tick. A quarter of the
# pixels buys the full 25 Hz at the same memory cost, and matches the recorded frame size, so the
# text is no softer than it was when a 1920 canvas was being downsampled into a 720p movie.
DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
WIDTH = 1280
HEIGHT = 720
SCALE = WIDTH / DESIGN_WIDTH

# Two plot cards across the top, a statistics card in the lower left. The gaps are wide enough
# that the body of the car reads through the middle of the frame.
HEAVE_CARD = (28, 28, 940, 424)
PITCH_CARD = (980, 28, 1892, 424)
PEDAL_CARD = (28, 432, 1892, 560)
STATS_CARD = (28, 560, 620, 1052)

CARD_TITLE_HEIGHT = 46
PLOT_INSET_LEFT = 104
PLOT_INSET_RIGHT = 40
PLOT_INSET_BOTTOM = 30

WINDOW_SECONDS = 6.0
PANEL_HZ = 25.0

# Distance between the top edge a call site asks for and the y that `drawText` wants. Measured on
# a rendered frame: the ink top lands 2 px below the requested y at size 22 and 3 px at size 32, so
# `drawText`'s y behaves as the top of the text box rather than as a baseline - whatever
# `docs/reference/display.md` says - and no offset is needed.
TEXT_TOP_OFFSET = 0.0

# Font handling, deliberately restricted. Webots' `WbDisplayFont::loadFace` appends the whole
# TrueType file to a buffer list on *every* `setFont` call and never releases it, so each call
# costs the size of the font file (115 KB for Lucida Console). Text is therefore drawn grouped
# by size, and the size only changes four times per frame.
FONT = "Lucida Console"
SIZE_SMALL = 22
SIZE_BODY = 26
SIZE_TITLE = 32
SIZE_HERO = 56

# Webots renders Lucida Console at roughly 0.82 em per character: `setFont` takes the character
# height, and the advance that comes out is wider than the nominal 0.6 em of the typeface. Only
# used for centring and for placing a unit label after a value, so an approximation is enough.
CHARACTER_WIDTH = 0.82

# Value ranges. These are *floors*: the axis may grow to fit the data but never shrinks, so a trace
# never appears to jump because the axis rescaled under it.
HEAVE_FLOOR_MM = 10.0
PITCH_FLOOR_DEG = 2.0
# Fraction of headroom above the largest value in the window, before rounding to a nice limit.
HEADROOM = 1.05


def _fmt(value: float, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _nice_limit(value: float) -> float:
    """Round a half-range up to a value a viewer can read at a glance.

    The axis only ever grows, so a continuous limit would leave the labels reading things like
    "20.2" and "10.1" for the whole clip. Rounding keeps them at "25" and "12.5" while preserving
    the property that matters: the scale never shrinks under a trace that is already drawn.
    """
    if value <= 0.0:
        return value
    magnitude = 10.0 ** math.floor(math.log10(value))
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value <= step * magnitude:
            return step * magnitude
    return 10.0 * magnitude


def text_width(text, size):
    return int(size * CHARACTER_WIDTH * len(str(text)))


class Hud:
    """Accumulates samples and paints the panel."""

    def __init__(self, display, *, fit_heave_mm, fit_pitch_deg, fit_speed_r2, source_note):
        self.display = display
        self.fit_heave_mm = fit_heave_mm
        self.fit_pitch_deg = fit_pitch_deg
        self.fit_speed_r2 = fit_speed_r2
        self.source_note = source_note

        # Ring buffers: (t, speed, heave_measured, pitch_measured, heave_model, pitch_model, contact)
        self.samples = []
        self.next_draw_time = -1.0
        # Grow-only axis limits.
        self.heave_limit = HEAVE_FLOOR_MM
        self.pitch_limit = PITCH_FLOOR_DEG
        self.setup()

    # -- Display helpers ----------------------------------------------------
    def setup(self):
        self._font_size = None
        self._pending_text = []
        self._font_calls = 0
        self._apply_font(SIZE_SMALL)

    def _apply_font(self, size):
        if size == self._font_size:
            return
        self.display.setFont(FONT, int(size), True)
        self._font_size = int(size)
        self._font_calls += 1

    def _color(self, value, alpha=1.0):
        self.display.setColor(value)
        self.display.setAlpha(alpha)

    def _rect(self, x, y, w, h, color, alpha=1.0):
        # The Display API takes integer pixels with (0,0) at the top left and requires a
        # strictly positive size, so the two edges are rounded independently: rounding the
        # origin and the extent separately turns a sub-pixel band into a null-width one,
        # which Webots rejects with "width argument is negative or null".
        left = int(math.floor(x * SCALE))
        top = int(math.floor(y * SCALE))
        self._color(color, alpha)
        self.display.fillRectangle(left, top,
                                   max(1, int(math.ceil((x + w) * SCALE)) - left),
                                   max(1, int(math.ceil((y + h) * SCALE)) - top))

    def _line(self, x1, y1, x2, y2, color, alpha=1.0):
        self._color(color, alpha)
        self.display.drawLine(int(round(x1 * SCALE)), int(round(y1 * SCALE)),
                              int(round(x2 * SCALE)), int(round(y2 * SCALE)))

    def _text(self, text, x, y, color, size=SIZE_BODY, alpha=1.0):
        # Queued rather than drawn: `draw()` flushes them grouped by size. `y` is the text's
        # top edge, which is the coordinate `drawText` takes as well. The design-space font size
        # becomes a canvas font size here, so nothing downstream needs to know about `SCALE`.
        self._pending_text.append((max(8, int(round(size * SCALE))), str(text),
                                   int(round(x * SCALE)), int(round(y * SCALE)), color, alpha))

    def _flush_text(self):
        for size, text, x, y, color, alpha in sorted(self._pending_text,
                                                     key=lambda item: item[0]):
            self._apply_font(size)
            self._color(color, alpha)
            self.display.drawText(text, x, y + int(round(size * TEXT_TOP_OFFSET)))
        self._pending_text.clear()

    # -- data ---------------------------------------------------------------
    def push(self, t, speed_kmh, heave_measured_mm, pitch_measured_deg,
             heave_model_mm, pitch_model_deg, contact_class, throttle=0.0, brake=0.0):
        """Record one sample. The model values are `None` while it is not being simulated."""
        self.samples.append((t, speed_kmh, heave_measured_mm, pitch_measured_deg,
                             heave_model_mm, pitch_model_deg, contact_class,
                             throttle, brake))
        # Keep a little more than the window, plus one sample of slack.
        cutoff = t - WINDOW_SECONDS * 1.5
        if self.samples and self.samples[0][0] < cutoff:
            drop = 0
            while drop < len(self.samples) - 2 and self.samples[drop][0] < cutoff:
                drop += 1
            del self.samples[:drop]
        self._grow_limits()

    def _grow_limits(self):
        window_start = self.samples[-1][0] - WINDOW_SECONDS if self.samples else 0.0
        for sample in self.samples:
            if sample[0] < window_start:
                continue
            for value in (sample[2], sample[4]):
                if value is not None and math.isfinite(value):
                    self.heave_limit = max(self.heave_limit,
                                           _nice_limit(abs(value) * HEADROOM))
            for value in (sample[3], sample[5]):
                if value is not None and math.isfinite(value):
                    self.pitch_limit = max(self.pitch_limit,
                                           _nice_limit(abs(value) * HEADROOM))

    # -- drawing ------------------------------------------------------------
    def maybe_draw(self, t_now, phase_name):
        """Paint at most `PANEL_HZ`, decoupled from the physics rate."""
        if t_now < self.next_draw_time:
            return False
        self.next_draw_time = t_now + 1.0 / PANEL_HZ
        self.draw(t_now, phase_name)
        return True

    def draw(self, t_now, phase_name):
        self._clear()
        self._plot_card(HEAVE_CARD, "HEAVE (mm)", t_now, 2, 4, self.heave_limit, 1)
        self._plot_card(PITCH_CARD, "PITCH (deg)", t_now, 3, 5, self.pitch_limit, 2)
        self._pedal_card(t_now)
        self._stats_card(t_now, phase_name)
        self._flush_text()

    def _clear(self):
        # Alpha 0 paints nothing into the colour channels but sets every pixel's alpha to 0, so
        # whatever the cards leave untouched shows the simulation. This also switches the
        # Display's texture to a blended one, which is what makes the gaps work at all.
        self.display.setAlpha(0.0)
        self.display.fillRectangle(0, 0, WIDTH, HEIGHT)
        self.display.setAlpha(1.0)

    def _card(self, card, alpha=0.90):
        x0, y0, x1, y1 = card
        # A soft dark edge underneath reads as a shadow and keeps the card from looking pasted on
        # top of a bright sky.
        self._rect(x0 + 5, y0 + 7, x1 - x0, y1 - y0, SHADOW, 0.30)
        self._rect(x0, y0, x1 - x0, y1 - y0, PANEL_BORDER, alpha)
        self._rect(x0 + 3, y0 + 3, x1 - x0 - 6, y1 - y0 - 6, PANEL_BACKGROUND, alpha)

    def _plot_card(self, card, title, t_now, measured_index, model_index, limit, digits):
        x0, y0, x1, y1 = card
        self._card(card)
        # Title row: the name on the left, the axis half-range on the right. Keeping both out
        # of the plot means the interior labels have the panel to themselves.
        self._text(title, x0 + 30, y0 + 14, MEASURED, SIZE_TITLE)
        self._text(f"+/-{limit:g}", x1 - 170, y0 + 22, MUTED, SIZE_BODY)

        left = x0 + PLOT_INSET_LEFT
        right = x1 - PLOT_INSET_RIGHT
        top = y0 + CARD_TITLE_HEIGHT + 34
        bottom = y1 - PLOT_INSET_BOTTOM
        height = bottom - top

        # Three interior gridlines: enough to read a magnitude, few enough to stay legible when
        # the panel is small on screen. The extremes carry their value on the left already.
        for fraction in (0.0, 0.5, 1.0):
            y = bottom - fraction * height
            value = (fraction * 2.0 - 1.0) * limit
            self._line(left, y, right, y, PANEL_BORDER if fraction == 0.5 else GRID)
            self._text(_fmt(value, digits), x0 + 26, y - 14, MUTED, SIZE_SMALL)

        window_start = t_now - WINDOW_SECONDS
        for second in range(int(math.ceil(window_start)), int(math.floor(t_now)) + 1):
            x = self._x(second, window_start, t_now, left, right)
            if left <= x <= right:
                self._line(x, top, x, bottom, GRID)

        self._plot(measured_index, MEASURED, 3, window_start, t_now, top, bottom, limit,
                   left, right)
        self._plot(model_index, MODEL, 7, window_start, t_now, top, bottom, limit, left, right)

        # Readhead and current-value dots at the right edge.
        self._line(right, top, right, bottom, MODEL, 0.35)
        latest = self.samples[-1] if self.samples else None
        if latest is not None:
            for index, color in ((measured_index, MEASURED), (model_index, MODEL)):
                value = latest[index]
                if value is None or not math.isfinite(value):
                    continue
                y = self._y(value, limit, top, bottom)
                self._color(color)
                radius = max(3, int(round(9 * SCALE)))
                self.display.fillOval(int(round(right * SCALE)), int(round(y * SCALE)),
                                      radius, radius)

    def _x(self, value, window_start, t_now, left, right):
        span = max(t_now - window_start, 1e-6)
        return left + (value - window_start) / span * (right - left)

    def _y(self, value, limit, top, bottom):
        height = bottom - top
        clipped = max(-limit, min(limit, value))
        return bottom - (clipped + limit) / (2.0 * limit) * height

    def _plot(self, index, color, linewidth, window_start, t_now, top, bottom, limit,
              left, right):
        # Decimate to roughly one point per 2 px: the curve looks identical and the draw cost stays
        # bounded no matter how many samples the window holds.
        visible = [sample for sample in self.samples if sample[0] >= window_start]
        if len(visible) < 2:
            return
        step = max(1, len(visible) // 400)
        points = []
        for sample in visible[::step]:
            value = sample[index]
            if value is None or not math.isfinite(value):
                # A break in the model series: flush what we have and start a new stroke, so a
                # lift-off shows as a genuine gap rather than a straight line across it.
                self._stroke(points, color, linewidth, window_start, t_now, top, bottom, limit,
                             left, right)
                points = []
                continue
            points.append((sample[0], value))
        self._stroke(points, color, linewidth, window_start, t_now, top, bottom, limit,
                     left, right)

    def _stroke(self, points, color, linewidth, window_start, t_now, top, bottom, limit,
                left, right):
        if len(points) < 2:
            return
        self._color(color)
        fill = getattr(self.display, "fillPolygon", None)
        if fill is not None:
            # One filled polygon instead of one command per pixel row per segment. The trace is a
            # function of time, so its band is the upper edge walked forwards and the lower edge
            # walked back: 2n vertices, one command. Emulating the width with parallel 1 px lines
            # instead costs `linewidth` commands per segment - four thousand per frame here - and
            # Webots' display path allocates per command, which is enough to exhaust memory over a
            # 23 s clip.
            # `half` is in canvas pixels: the stroke widths are design-space, so the band is
            # converted first and then offset, rather than mixing the two units.
            half = max(1, int(round((linewidth - 1) * 0.5 * SCALE)))
            xs = []
            ys = []
            for t, value in points:
                xs.append(int(round(self._x(t, window_start, t_now, left, right) * SCALE)))
                ys.append(int(round(self._y(value, limit, top, bottom) * SCALE)) - half)
            for t, value in reversed(points):
                xs.append(int(round(self._x(t, window_start, t_now, left, right) * SCALE)))
                ys.append(int(round(self._y(value, limit, top, bottom) * SCALE)) + half)
            fill(xs, ys)
            return
        thick = getattr(self.display, "drawThickLine", None)
        if thick is not None:
            for index in range(len(points) - 1):
                t0, v0 = points[index]
                t1, v1 = points[index + 1]
                thick(self._x(t0, window_start, t_now, left, right),
                      self._y(v0, limit, top, bottom),
                      self._x(t1, window_start, t_now, left, right),
                      self._y(v1, limit, top, bottom), linewidth)
            return
        for offset in range(linewidth):
            for index in range(len(points) - 1):
                t0, v0 = points[index]
                t1, v1 = points[index + 1]
                self._line(self._x(t0, window_start, t_now, left, right),
                           self._y(v0, limit, top, bottom) + offset,
                           self._x(t1, window_start, t_now, left, right),
                           self._y(v1, limit, top, bottom) + offset,
                           color)

    def _pedal_card(self, t_now):
        """Throttle up from the centre line, brake down, on the plots' time axis.

        The pedals are the cause of everything in the two cards above, so they are drawn on the
        same window and the same horizontal scale: a throttle step lines up vertically with the
        heave and pitch it produces.
        """
        x0, y0, x1, y1 = PEDAL_CARD
        self._card(PEDAL_CARD)
        self._text("PEDAL INPUT", x0 + 20, y0 + 6, MUTED, SIZE_SMALL)

        latest = self.samples[-1] if self.samples else None
        throttle = latest[7] if latest else 0.0
        brake = latest[8] if latest else 0.0
        # Right-aligned as a pair so the two legends never collide whatever the percentages are.
        brake_text = f"BRAKE {brake * 100:3.0f} %"
        brake_x = x1 - 26 - text_width(brake_text, SIZE_SMALL)
        self._rect(brake_x - 34, y0 + 14, 22, 12, BRAKE)
        self._text(brake_text, brake_x, y0 + 6, MEASURED, SIZE_SMALL)
        throttle_text = f"THROTTLE {throttle * 100:3.0f} %"
        throttle_x = brake_x - 76 - text_width(throttle_text, SIZE_SMALL)
        self._rect(throttle_x - 34, y0 + 14, 22, 12, THROTTLE)
        self._text(throttle_text, throttle_x, y0 + 6, MEASURED, SIZE_SMALL)

        left = HEAVE_CARD[0] + PLOT_INSET_LEFT
        right = PITCH_CARD[2] - PLOT_INSET_RIGHT
        centre = y0 + 64
        half = 32
        self._line(left, centre, right, centre, PANEL_BORDER)

        window_start = t_now - WINDOW_SECONDS
        visible = [sample for sample in self.samples if sample[0] >= window_start]
        # The pedals are held constant between samples, so consecutive samples with the same value
        # collapse into one rectangle. Without that this strip costs a fillRectangle per sample per
        # pedal - two and a half thousand commands a frame - for a shape that only ever has a
        # handful of corners.
        for value_index, colour, direction in ((7, THROTTLE, -1), (8, BRAKE, +1)):
            run_start = None
            run_end = None
            run_value = 0.0
            # The trailing `None` flushes the last run; it must not be indexed.
            for sample in visible + [None]:
                value = None if sample is None else (sample[value_index] or 0.0)
                if run_start is not None and value is not None and value == run_value:
                    run_end = sample[0]
                    continue
                if run_start is not None and run_value > 0.0:
                    x = self._x(run_start, window_start, t_now, left, right)
                    width = max(2, int(self._x(run_end, window_start, t_now, left, right))
                                - int(x) + 1)
                    height = max(2, int(round(min(1.0, run_value) * half)))
                    top = centre - height if direction < 0 else centre
                    self._rect(x, top, width, height, colour, 0.95)
                if sample is None:
                    break
                run_start = sample[0]
                run_end = sample[0]
                run_value = value

    def _stats_card(self, t_now, phase_name):
        x0, y0, x1, y1 = STATS_CARD
        self._card(STATS_CARD)
        width = x1 - x0

        self._text("HALF-CAR MODEL", x0 + 28, y0 + 16, MEASURED, SIZE_TITLE)
        self._text("vs WEBOTS", x0 + 28, y0 + 56, MODEL, SIZE_TITLE)
        # The phase callout is load-bearing: during the pulse train it is what explains why the body
        # is rocking, for a viewer watching muted.
        label = phase_name.replace("_", " ").upper() if phase_name else ""
        self._text(label, x0 + 28, y0 + 100, MUTED, SIZE_BODY)
        self._line(x0 + 28, y0 + 140, x1 - 28, y0 + 140, PANEL_BORDER)

        latest = self.samples[-1] if self.samples else None
        speed = latest[1] if latest else float("nan")
        heave = latest[2] if latest else float("nan")
        pitch = latest[3] if latest else float("nan")

        rows = [("SPEED", _fmt(speed, 0), "km/h"),
                ("HEAVE", _fmt(heave, 1), "mm"),
                ("PITCH", _fmt(pitch, 2), "deg")]
        for index, (label, value, unit) in enumerate(rows):
            y = y0 + 156 + index * 82
            self._text(label, x0 + 28, y + 18, MUTED, SIZE_SMALL)
            self._text(value, x0 + 148, y, MEASURED, SIZE_HERO)
            self._text(unit, x0 + 166 + text_width(value, SIZE_HERO), y + 24, MUTED, SIZE_BODY)

        # Contact state as a chip rather than a timeline: with the schedule tuned to keep the tyres
        # down, the interesting claim is that it is grounded, and that has to be stated outright.
        contact = (latest[6] if latest else None) or "none"
        colours = {"none": GROUNDED, "front": LIFT_FRONT, "rear": LIFT_REAR,
                   "all": LIFT_ALL, "settling": PANEL_BORDER}
        captions = {"none": "ALL FOUR WHEELS PLANTED", "front": "FRONT AXLE LIFTED",
                    "rear": "REAR AXLE LIFTED", "all": "ALL WHEELS LIFTED",
                    "settling": "SETTLING"}
        chip = y1 - 72
        self._rect(x0 + 28, chip, width - 56, 44, colours.get(contact, PANEL_BORDER), 0.88)
        self._text(captions.get(contact, contact.upper()), x0 + 44, chip + 8,
                   PANEL_BACKGROUND if contact == "none" else MEASURED, SIZE_BODY)
