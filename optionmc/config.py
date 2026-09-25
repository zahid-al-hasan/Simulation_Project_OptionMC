from pricing import OptionPricing

N_PATHS = 512
SEED = 45

options = ["call", "put"]

pricing_modes = {
    "standard" : OptionPricing.standard_mc,
    "antithetic" : OptionPricing.antithetic_mc,
    "ctrl_var" : OptionPricing.control_variate_mc,
    "stratified" : OptionPricing.stratified_mc,
    "quasi" : OptionPricing.quasi_mc
}

Z_REF = 1.96