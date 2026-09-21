"""Actuator catalogue - datasheet numbers, one place, nothing invented here.

Every row is transcribed from the vendor product page; `step` is the vendor
STEP file downloaded from step.parts and checksum-verified, and `size_mm` was
measured back off that file (probe_motor) to confirm it matches the page.

Sources:
  AK80-64  https://www.cubemars.com/product/ak80-64-kv80-robotic-actuator.html
  AK60-39  https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html
  AK45-36  https://www.cubemars.com/product/AK45-36.html
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
}

RPM = 3.141592653589793 / 30.0


def rad_s(name, which="rated_rpm"):
    return MOTORS[name][which] * RPM
