class Postprocessing:
    def postprocessing_(sim_output):
        sim_output["micro-scalar-data"] = sim_output["micro-scalar-data"] + 20
        return sim_output
