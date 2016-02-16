import os                                   # system functions
import nipype.interfaces.fsl as fsl         # fsl
import nipype.interfaces.afni as afni       # afni
import nipype.interfaces.ants as ants       # ants
import nipype.pipeline.engine as pe         # the workflow and node wrappers
import nipype.interfaces.io as nio          # Input/Output
import nipype.interfaces.utility as util    # utility
import time                                 # Time used to measure length of pipeline

#======================================================================
# 1. Variable Specification
#======================================================================

tic = time.clock()
experiment_dir = '/Users/duncanlab/Documents/People/Shafquat/'             # location of data folder

# Count all the subfolders within a given directory
subs = next(os.walk(experiment_dir+'/Subjects'))[1]
subject_list = [] # Initialize an empty list to store subjects
session_list = ['Run1', 'Run2'] #, 'Run3', 'Run4', 'Run5']              # list of session identifiers

# list of subject identifiers
for subject in subs:
    subject_list.append(subject)

output_dir = 'output_firstSteps'          # name of output folder
working_dir = 'workingdir_firstSteps'     # name of working directory

number_of_slices = 40                     # number of slices in volume
TR = 2.0                                  # time repetition of volume
smoothing_size = 6                        # size of FWHM in mm
number_volumes_trim = 8                   # size of FWHM in mm

#======================================================================
# 2. NODE SPECIFICATION
#======================================================================

volTrim = pe.Node(fsl.ExtractROI(t_min = number_volumes_trim,
                                 t_size=-1),
                  name='volTrim')

sliceTiming = pe.Node(fsl.SliceTimer(time_repetition = TR,
                                     interleaved = True),
                   name="sliceTiming")

motionCorrection = pe.Node(afni.Volreg(md1d_file = 'max_disp.1d',
                                       oned_file = 'mot_par.1d',
                                       zpad = 3,
                                       outputtype = 'NIFTI_GZ'),
                                       name="motionCorrection")

susan = pe.Node(fsl.SUSAN(brightness_threshold = 150,
                               fwhm = 3),
                     name="susan")

skullStrip = pe.Node(afni.SkullStrip(args = '-niter 400 -push_to_edge -fill_hole -touchup -touchup',
                    outputtype = 'NIFTI_GZ'),
                    name="skullStrip")

betFunc = pe.Node(fsl.BET(frac = 0.5,
                    vertical_gradient = 0),
                    name="betFunc")

tstat = pe.Node(afni.TStat(args = '-mean',
                           outputtype = 'NIFTI_GZ'),
                name="tstat")

mean2anat = pe.Node(fsl.FLIRT(dof = 6),
                    name='mean2anat')


#======================================================================
# 3. SPECIFY WORKFLOWS
#======================================================================
# Functional Workflow
preproc = pe.Workflow(name='preproc')
preproc.base_dir = os.path.join(experiment_dir, working_dir)

# Anatomical Workflow
preprocAnat = pe.Workflow(name='anatpreproc')
preprocAnat.base_dir = os.path.join(experiment_dir, working_dir)


#======================================================================
# 3. INPUT/OUTPUT SPECIFICATION (func)
#======================================================================

# Infosource - a function free node to iterate over the list of subject names
infosource = pe.Node(util.IdentityInterface(fields=['subject_id',
                                            'session_id']),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list),
                        ('session_id', session_list)]

# SelectFiles for Input
templates = {'anat': experiment_dir + '/Subjects/{subject_id}/Anatomical/*.nii.gz',
             'func': experiment_dir + '/Subjects/{subject_id}/{session_id}/{session_id}.nii.gz'}


selectfiles = pe.Node(nio.SelectFiles(templates), name="selectfiles")

# DataSink for Output
datasink = pe.Node(nio.DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")


# Use the following DataSink output substitutions
substitutions = [('_subject_id', ''),
                 ('_session_id_', '')]
datasink.inputs.substitutions = substitutions


#======================================================================
# 3. INPUT/OUTPUT SPECIFICATION (anat)
#======================================================================

# Infosource - a function free node to iterate over the list of subject names
infosourceAnat = pe.Node(util.IdentityInterface(fields=['subject_id']),
                  name="infosource")

infosourceAnat.iterables = [('subject_id', subject_list)]

# SelectFiles for Input
templatesAnat = {'anat': experiment_dir + '/Subjects/{subject_id}/Anatomical/*.nii.gz'}


selectfilesAnat = pe.Node(nio.SelectFiles(templatesAnat), name="selectfilesAnat")

# DataSink for Anatomical Output
datasinkAnat = pe.Node(nio.DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")


# Use the following DataSink output substitutions
substitutions = [('_subject_id_', '')]
datasinkAnat.inputs.substitutions = substitutions


#======================================================================
# 4. Motion Correction
#======================================================================

# Set up an extract node to extract the last volume from the last run
extract_ref = pe.Node(interface=fsl.ExtractROI(t_size=1),
                      name = 'extractref')

# Pick the last file from the list of files
def  picklast(path_to_run):
    import os

    # get path to subject
    subject = os.path.dirname(os.path.dirname(path_to_run))
    # all runs in subject path stored in a list
    runs = next(os.walk(subject))[1]
    # Get last run
    last_run = str(runs[-1])
    selected_run = subject + "/" + last_run + "/" + last_run + ".nii.gz"

    return selected_run


preproc.connect(selectfiles, ('func', picklast), extract_ref, 'in_file')


# Pick the last volume from the given run
def getlastvolume(func):
    from nibabel import load
    funcfile = func
    _,_,_,timepoints = load(funcfile).get_shape()
    # To return middle volume use (timepoints/2)-1
    return (timepoints)

preproc.connect(sliceTiming, ('slice_time_corrected_file', getlastvolume), extract_ref, 't_min')

# Take the extracted last volume from the last run and use it as a reference file
preproc.connect([(sliceTiming, motionCorrection, [('slice_time_corrected_file', 'in_file')]),
                 (extract_ref, motionCorrection, [('roi_file', 'basefile')])
                 ])

#======================================================================
# 5. Remove skull from anatomical image and automask the functional image
#======================================================================



# Connect the mean last run with skull stripping
# preproc.connect([(tstat, func_skullStrip, [('out_file','in_file')])])

preprocAnat.connect([(infosourceAnat, selectfilesAnat, [('subject_id', 'subject_id')]),
                     (selectfilesAnat, susan, [('anat', 'in_file')]),
                     (susan, skullStrip, [('smoothed_file', 'in_file')]),
                    (skullStrip, datasinkAnat, [('out_file', 'skull_stripped')])])

#======================================================================
# 6. Connecting Workflows to datasink
#======================================================================

# Connect all components of the preprocessing workflow
# Connect SelectFiles and DataSink to the workflow
preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                            ('session_id', 'session_id')]),
                 (selectfiles, volTrim, [('func', 'in_file')]),
                 (volTrim, sliceTiming, [('roi_file', 'in_file')]),
                 (volTrim, datasink, [('roi_file', 'trimmed')]),
                 (sliceTiming, datasink, [('slice_time_corrected_file', 'sliced')]),
                 (motionCorrection, datasink, [('out_file', 'motion_correct')]),
                 (motionCorrection, datasink, [('md1d_file', 'mc_md1d')]),
                 (motionCorrection, datasink, [('oned_file', 'mc_oned')]),
                                  ])


#======================================================================
# 6.5 New preprocess pipeline to retrieve motion corrected files
#======================================================================

# Functional Workflow
preproc2 = pe.Workflow(name='preproc2')
preproc2.base_dir = os.path.join(experiment_dir, working_dir)

# Infosource - a function free node to iterate over the list of subject names
infosource2 = pe.Node(util.IdentityInterface(fields=['subject_id',
                                            'session_id']),
                  name="infosource2")

infosource2.iterables = [('subject_id', subject_list)]

# Select the last run for each subject
templates2 = {'func2': experiment_dir + 'output_firstSteps/motion_correct/'+ session_list[-1] + '_{subject_id}/*.nii.gz'}

selectfiles2 = pe.Node(nio.SelectFiles(templates2), name="selectfiles2")

# Set up a extract node for the last run after motion control and get the mean
preproc2.connect([(infosource2, selectfiles2, [('subject_id', 'subject_id'),
                                            ('session_id', 'session_id')]),
                  (selectfiles2, tstat, [('func2', 'in_file')]),
                  (tstat, betFunc, [('out_file', 'in_file')]),
                  ])

preproc2.connect([(tstat, datasink, [('out_file', 'tstat')]),
                  (betFunc, datasink, [('out_file', 'betFunc')]),
                   ])


# Use the following DataSink output substitutions
substitutions = [('_subject_id_', '')]
datasinkAnat.inputs.substitutions = substitutions

#======================================================================
# 7. Connect Anatomical to Mean Functional image
#======================================================================

# Infosource - a function free node to iterate over the list of subject names
infosource3 = pe.Node(util.IdentityInterface(fields=['subject_id',
                                            'session_id']),
                  name="infosource3")

infosource3.iterables = [('subject_id', subject_list)]

# Select the Skull Stripped Anatomical image for each subject
templates3 = {'func3': experiment_dir + 'output_firstSteps/sill_stripped/_{subject_id}/*.nii.gz'}

selectfiles3 = pe.Node(nio.SelectFiles(templates3), name="selectfiles3")

preproc2.connect([(infosource3, selectfiles3, [('subject_id', 'subject_id'),
                                            ('session_id', 'session_id')]),
                  (selectfiles3, mean2anat, [('func3', 'reference')]),
                  (betFunc, mean2anat, [('out_file', 'in_file')]),
                  (mean2anat, datasink, [('out_file', 'mean2anat')]),
                   ])

#======================================================================
# 8. Run, Forrest, Run!
#======================================================================

# Run the Nodes
preprocAnat.run('MultiProc', plugin_args={'n_procs': 3})
preproc.run('MultiProc', plugin_args={'n_procs': 3})
#preproc2.run('MultiProc', plugin_args={'n_procs': 3})

toc = time.clock()
print("This process took " + str(toc-tic) + " minutes")

#======================================================================
# . Return output to subject directories
#======================================================================
"""
# get all the subfolders within a given directory
num_folders = next(os.walk(experiment_dir+'/output_firstSteps/final_output'))[1]

# move file
for folder in num_folders:
    dir_files = os.listdir(experiment_dir+'/output_firstSteps/final_output/'+folder+'/')
    for file_name in dir_files:
        full_file_name = experiment_dir+'/output_firstSteps/final_output/'+folder+'/' + '/' + file_name
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, experiment_dir+'/Subjects/'+folder[-6:]+'/'+folder[0:4]+'/')
"""
