import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from lib import bear

def run_post_processing(path):
    #load the retrieval configuration file
    model_config = bear.Config(path)

    # #create a pyBeAR retrieval object
    # model = bear.Retrieval(model_config)

    # print("Starting retrieval\n")
    # model.run()

    #create a pyBeAR retrieval post process object
    post_process = bear.PostProcess(model_config)

    print("Starting post process\n")
    post_process.run()


#setting the basic properties of the model
# retrieval_folder = "TransmissionExample/"
# retrieval_folder = "/work2/lbuc/lukas/Projects/TOI-270d_JWST/retrievals/TOI-270_oneNIRISS/mols_fullspectrum"
# retrieval_folder = "/work2/lbuc/lukas/Projects/Proposals/HD219134bc_rockies/mock_retrievals/HD219134b_CO2_1bar"
# retrieval_folder = "/work2/lbuc/lukas/Projects/Proposals/HD219134bc_rockies/mock_retrievals/HD219134c_CO2_1bar"
# retrieval_folder = "/work2/lbuc/lukas/Projects/Proposals/HD219134bc_rockies/mock_retrievals/HD219134b_H2O_1bar"
# retrieval_folder = "/work2/lbuc/lukas/Projects/Proposals/HD219134bc_rockies/mock_retrievals/HD219134c_H2O_1bar"
retrieval_folder = "/work2/lbuc/lukas/Projects/CH_allspecies/TOI-270_sample_CH4only"

run_post_processing(retrieval_folder)
