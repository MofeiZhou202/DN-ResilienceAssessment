def case24_failrate():
    """
    Provides failure rate data for the 24-bus test system.

    Returns:
        dict: Dictionary containing generation and branch failure rate data
    """
    failrate = {}

    # Generator MTTF (hours)
    failrate['genmttf'] = [
        450, 450, 1960, 1960, 450,
        450, 1960, 1960, 1200, 1200,
        1200, 950, 950, 950, 10000,  # 10000 bus 14 synchronous compensator
        2940, 2940, 2940, 2940, 2940,
        960, 960, 1100, 1100, 1980,
        1980, 1980, 1980, 1980, 1980,
        960, 960, 1150
    ]

    # Generator MTTR (hours)
    failrate['genmttr'] = [
        50, 50, 40, 40, 50,
        50, 40, 40, 50, 50,
        50, 50, 50, 50, 0.1,  # 0.1 bus 14 synchronous compensator
        60, 60, 60, 60, 60,
        40, 40, 150, 150, 20,
        20, 20, 20, 20, 20,
        40, 40, 100
    ]

    # Generator scheduled maintenance weeks (weeks/yr)
    failrate['genweeks'] = [
        2, 2, 3, 3, 2,
        2, 3, 3, 3, 3,
        3, 4, 4, 4, 0.1,  # 0.1 bus 14 synchronous compensator
        2, 2, 2, 2, 2,
        4, 4, 6, 6, 2,
        2, 2, 2, 2, 2,
        4, 4, 5
    ]

    # Branch force outage rate (1/yr)
    failrate['brlambda'] = [
        0.24, 0.51, 0.33, 0.39, 0.48, 0.38,
        0.02, 0.36, 0.34, 0.33, 0.30, 0.44,
        0.44, 0.02, 0.02, 0.02, 0.02, 0.40,
        0.39, 0.40, 0.52, 0.49, 0.38, 0.33,
        0.41, 0.41, 0.41, 0.35, 0.34, 0.32,
        0.54, 0.35, 0.35, 0.38, 0.38, 0.34,
        0.34, 0.45
    ]

    # Branch outage durations (hours)
    failrate['brdur'] = [
        16, 10, 10, 10, 10, 768, 10, 10, 35, 10, 10, 10,
        10, 768, 768, 768, 768, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11
    ]

    return failrate


def failprob():
    """
    Computes failure probabilities for generators and branches.

    Returns:
        dict: Dictionary containing generator and branch failure probabilities
    """
    # Get failure rate data
    failrate = case24_failrate()

    # Compute the failure probability of generators
    probgen = [mttr / (mttf + mttr) for mttf, mttr in zip(failrate['genmttf'], failrate['genmttr'])]

    # Alternative calculation considering planned maintenance (commented out)
    '''
    # When taking considerations of planned maintenance
    weekhours = 7 * 24

    mttrp = [genweek * weekhours for genweek in failrate['genweeks']]
    genmiup = [8760 / mttr for mttr in mttrp]
    mttfp = [8760 - mttr for mttr in mttrp]
    genlambdap = [8760 / mttf for mttf in mttfp]

    # Component of forced outage
    genmiu = [8760 / mttr for mttr in failrate['genmttr']]
    genlambda = [8760 / mttf for mttf in failrate['genmttf']]

    # Combination of two parts
    probgen = [(lamb * miup + lambp * miu) / (lamb * miup + lambp * miu + miup * miu) 
              for lamb, miup, lambp, miu in zip(genlambda, genmiup, genlambdap, genmiu)]
    '''

    # Branch data
    brmiu = [8760 / dur for dur in failrate['brdur']]
    probbr = [lamb / (lamb + miu) for lamb, miu in zip(failrate['brlambda'], brmiu)]

    # Combination matrix
    totalprob = {
        'probgen': probgen,
        'probbr': probbr
    }

    return totalprob


# Example usage:
if __name__ == "__main__":
    # Calculate failure probabilities
    probs = failprob()

    # Print results
    print("Generator failure probabilities:", probs['probgen'])
    print("Branch failure probabilities:", probs['probbr'])