"""
Post-processing
In this script a post-processing step is defined.
A script like this can be used to post-process the simulation output before writing it to a file,
if this is not done in the micro simulation itself.
"""


class Postprocessing:
    def postprocessing(sim_output):
        """Post-process the simulation output.

        Parameters
        ----------
        sim_output : dict
            Raw simulation output.

        Returns
        -------
        sim_output : dict
            Post-processed simulation output.
        """
        sim_output["micro-scalar-data"] = sim_output["micro-scalar-data"] + 20
        return sim_output
