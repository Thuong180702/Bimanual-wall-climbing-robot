"""Actuator catalogue - datasheet numbers, one place, nothing invented here.

Every row is transcribed from the vendor product page; `step` is the vendor
STEP file downloaded from step.parts and checksum-verified, and `size_mm` was
measured back off that file (probe_motor) to confirm it matches the page.

Sources:
  AK80-64  https://www.cubemars.com/product/ak80-64-kv80-robotic-actuator.html
  AK60-39  https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html
  AK45-36  https://www.cubemars.com/product/AK45-36.html
  RS04     RobStride 04 (the full robot's hip pitch and knee): Seeed Studio's product
           page https://www.seeedstudio.com/Robostride-04-Actuator-p-6775.html
           (120 N.m peak, 40 N.m rated at 50 rpm, 200 rpm no-load, 9:1, 48 V,
           1.42 kg); the STEP is step.parts robstride_04, and size_mm is off it.
  X12-320  MyActuator RMD-X12-P20-320 (the full robot's shoulders, elbows and wrist
           flexion): MyActuator's own 2D drawing and table, myactuator.pages.dev/downloads/
           rmd-x12-p20-320-2d-drawing.pdf (Ø124 x 85, 2.37 kg, 48 V, 20:1, 85 N.m rated at
           100 rpm, 320 peak, 125 rpm no-load); the STEP is step.parts myactuator_rmd_x12_320.
  X15-450  MyActuator RMD-X15-P20-450 (the full robot's elbows): its 2D drawing
           (myactuator.pages.dev/downloads/rmd-x15-p20-450-2d-drawing.pdf): Ø166 x 69 body
           + a 21 mm Ø124 connector hub behind it, 3.5 kg, 72 V, 20.25:1, 145 N.m rated at
           98 rpm, 450 peak, 108 rpm no-load. Runs on 72 V, not the 48 V the rest use.
  AKE90-8  CubeMars AKE90-8 KV35 (the full robot's wrist deviation): CubeMars' parameter
           sheet img.cubemars.com/products/cubemars-product-parameter/AKE90-8-KV35.pdf
           (Ø107.5 x 43.5, 1400 g, 48 V, 8:1, 55 N.m rated at 120 rpm, 170 peak, 210 rpm
           no-load); the STEP is step.parts cubemars_ake90_8_kv35. It is the motor and
           gearbox only (U/V/W leads, a Ø6 encoder magnet at the back): a CubeMars driver
           board (the AK series' CAN protocol) bolts behind it. The board is not in the
           STEP; mass_kg carries +50 g and IFACE/_envelope an 11 mm Ø80 allowance for it
           (my estimate, to replace with the board's drawing).
"""


MOTORS = {
    "AK80-64": dict(
        vendor="CubeMars", model="AK80-64 KV80",
        step="cubemars_ak80_64_kv80.step",
        peak_Nm=120.0, rated_Nm=48.0,
        rated_rpm=48.0, noload_rpm=75.0,          # at 48 V
        mass_kg=0.850, dia_mm=98.0, len_mm=61.9,
        ratio=64.0, kt=0.136, rated_A=7.0, peak_A=19.0, volt="24/48",
        backlash_deg=0.18, enc_bits=14,
        size_mm=(61.9, 98.0, 98.0)),                # measured off the STEP
    "AK60-39": dict(
        vendor="CubeMars", model="AK60-39 V3.0 KV80",
        step="cubemars_ak60_39_v3_kv80.step",
        peak_Nm=72.0, rated_Nm=24.0,
        rated_rpm=70.0, noload_rpm=98.0,           # at 48 V
        mass_kg=0.750, dia_mm=79.0, len_mm=67.0,
        ratio=39.0, kt=0.12, rated_A=4.5, peak_A=17.0, volt="48",
        backlash_arcmin=15, enc_bits=21,
        size_mm=(79.0, 79.0, 67.0)),
    "AK45-36": dict(
        vendor="CubeMars", model="AK45-36 KV80",
        step="cubemars_ak45_36_kv80.step",
        peak_Nm=24.0, rated_Nm=8.0,
        rated_rpm=40.0, noload_rpm=52.0,           # at 24 V
        mass_kg=0.340, dia_mm=55.0, len_mm=54.0,
        ratio=36.0, kt=0.11, rated_A=2.0, peak_A=6.5, volt="24",
        backlash_arcmin=12, enc_bits=14,
        size_mm=(55.0, 54.0, 55.0)),
    "RS04": dict(
        vendor="RobStride", model="RobStride 04",
        step="robstride_04.step",
        peak_Nm=120.0, rated_Nm=40.0,
        rated_rpm=50.0, noload_rpm=200.0,          # at 48 V
        mass_kg=1.42, dia_mm=120.0, len_mm=55.7,
        ratio=9.0, volt="48",
        size_mm=(120.0, 120.0, 55.7),              # measured off the STEP
        proud_mm=3.0),                             # the carrier and its dowels stand
                                                   # this far past the output face
    "X12-320": dict(
        vendor="MyActuator", model="RMD-X12-P20-320",
        step="myactuator_rmd_x12_320.step",
        peak_Nm=320.0, rated_Nm=85.0,
        rated_rpm=100.0, noload_rpm=125.0,         # at 48 V
        mass_kg=2.37, dia_mm=124.0, len_mm=85.0,
        ratio=20.0, volt="48",
        size_mm=(85.0, 124.0, 124.0)),             # measured off the STEP
    "X15-450": dict(
        vendor="MyActuator", model="RMD-X15-P20-450",
        step="myactuator_rmd_x15_450.step",
        peak_Nm=450.0, rated_Nm=145.0,
        rated_rpm=98.0, noload_rpm=108.0,          # at 72 V
        mass_kg=3.5, dia_mm=166.0, len_mm=90.0,    # 69 body + the 21 mm connector hub
        ratio=20.25, volt="72",
        size_mm=(90.0, 166.0, 166.0)),             # measured off the STEP
    "AKE90-8": dict(
        vendor="CubeMars", model="AKE90-8 KV35",
        step="cubemars_ake90_8_kv35.step",
        peak_Nm=170.0, rated_Nm=55.0,
        rated_rpm=120.0, noload_rpm=210.0,         # at 48 V
        mass_kg=1.45, dia_mm=107.5, len_mm=43.5,   # 1400 g + ~50 g driver board (estimate)
        ratio=8.0, volt="48",
        size_mm=(107.5, 107.5, 43.5)),             # measured off the STEP
}

RPM = 3.141592653589793 / 30.0


def rad_s(name, which="rated_rpm"):
    return MOTORS[name][which] * RPM
