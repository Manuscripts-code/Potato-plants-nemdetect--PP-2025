import numpy as np


def get_labels_by_group(group: int) -> dict[str, list[int]]:
    well_water_pallida_low = [1, 2, 3, 4, 5, 6, 7]
    well_water_pallida_high = [8, 9, 10, 11, 12, 13, 14]
    well_water_rostoch_low = [15, 16, 17, 18, 19, 20, 21]
    well_water_rostoch_high = [22, 23, 24, 25, 26, 27, 28]
    well_water_control = [29, 30, 31, 32, 33, 34, 35]
    def_water_pallida_low = [36, 37, 38, 39, 40, 41, 42]
    def_water_pallida_high = [43, 44, 45, 46, 47, 48, 49]
    def_water_rostoch_low = [50, 51, 52, 53, 54, 55, 56]
    def_water_rostoch_high = [57, 58, 59, 60, 61, 62, 63]
    def_water_control = [64, 65, 66, 67, 68, 69, 70]

    items_list_c1 = []
    items_list_c2 = []
    items_list_c3 = []
    items_list_c4 = []
    items_list_c5 = []
    items_list_c6 = []
    items_list_c7 = []
    items_list_c8 = []
    items_list_c9 = []
    items_list_c10 = []

    ################

    # Controls

    if group == 0:
        items_list_c1 = well_water_control
        items_list_c2 = def_water_control

    # Differentiate between water treatment

    if group == 1:
        items_list_c1 = (
            well_water_pallida_low
            + well_water_pallida_high
            + well_water_rostoch_low
            + well_water_rostoch_high
            + well_water_control
        )
        items_list_c2 = (
            def_water_pallida_low
            + def_water_pallida_high
            + def_water_rostoch_low
            + def_water_rostoch_high
            + def_water_control
        )

    # Inoculation severity

    if group == 2:
        items_list_c1 = well_water_rostoch_high
        items_list_c2 = well_water_rostoch_low

    if group == 3:
        items_list_c1 = well_water_pallida_high
        items_list_c2 = well_water_pallida_low

    if group == 4:
        items_list_c1 = def_water_rostoch_low
        items_list_c2 = def_water_rostoch_high

    if group == 5:
        items_list_c1 = def_water_pallida_low
        items_list_c2 = def_water_pallida_high

    # Inoculation nematode type - multiple varieties combined

    if group == 6:
        items_list_c1 = well_water_pallida_low + well_water_pallida_high
        items_list_c2 = well_water_rostoch_low + well_water_rostoch_high

    if group == 7:
        items_list_c1 = def_water_pallida_low + def_water_pallida_high
        items_list_c2 = def_water_rostoch_low + def_water_rostoch_high

    # Inoculation nematode with control

    if group == 8:
        items_list_c1 = well_water_rostoch_low + well_water_rostoch_high
        items_list_c2 = well_water_control

    if group == 9:
        items_list_c1 = well_water_pallida_low + well_water_pallida_high
        items_list_c2 = well_water_control

    if group == 10:
        items_list_c1 = def_water_rostoch_low + def_water_rostoch_high
        items_list_c2 = def_water_control

    if group == 11:
        items_list_c1 = def_water_pallida_low + def_water_pallida_high
        items_list_c2 = def_water_control

    # Compare inoculation nematode type with control - combine upper two

    if group == 12:
        items_list_c1 = well_water_pallida_low + well_water_pallida_high
        items_list_c2 = well_water_rostoch_low + well_water_rostoch_high
        items_list_c3 = well_water_control

    if group == 13:
        items_list_c1 = def_water_pallida_low + def_water_pallida_high
        items_list_c2 = def_water_rostoch_low + def_water_rostoch_high
        items_list_c3 = def_water_control

    # Differentiate any nematode inoculation with control

    if group == 14:
        items_list_c1 = (
            well_water_rostoch_low
            + well_water_rostoch_high
            + well_water_pallida_low
            + well_water_pallida_high
        )
        items_list_c2 = well_water_control

    if group == 15:
        items_list_c1 = (
            def_water_rostoch_low
            + def_water_rostoch_high
            + def_water_pallida_low
            + def_water_pallida_high
        )
        items_list_c2 = def_water_control

    # Differentiate biotic vs abiotic stress

    if group == 16:
        items_list_c1 = (
            well_water_pallida_low
            + well_water_pallida_high
            + well_water_rostoch_low
            + well_water_rostoch_high
        )
        items_list_c2 = (
            def_water_pallida_low
            + def_water_pallida_high
            + def_water_rostoch_low
            + def_water_rostoch_high
        )
        items_list_c3 = well_water_control
        items_list_c4 = def_water_control

    if group == 17:
        items_list_c1 = (
            well_water_pallida_low
            + well_water_pallida_high
            + well_water_rostoch_low
            + well_water_rostoch_high
        )
        items_list_c2 = well_water_control
        items_list_c3 = def_water_control

    categories = [
        items_list_c1,
        items_list_c2,
        items_list_c3,
        items_list_c4,
        items_list_c5,
        items_list_c6,
        items_list_c7,
        items_list_c8,
        items_list_c9,
        items_list_c10,
    ]
    return {f"c{idx+1}": item for idx, item in enumerate(categories) if item}


def count_unique_labels(labels: np.ndarray) -> dict:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_labels, counts))
