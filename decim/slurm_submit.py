
#!/usr/bin/env python
'''
Parallelize functions with simple arguments via SLURM
'''

import subprocess
import tempfile
import os
import errno


def submit(walltime, memory, tmpdir, logdir, workdir, script, name,
           nodes=1, tasks=16,
           shellfname=None):
    '''
    Submit a script to torque
    '''
    print('script in submit {}'.format(script))
    sbatch_directives = '''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={tasks}
#SBATCH --time={walltime}
#SBATCH --export=NONE
#SBATCH --mem={memory}GB
#SBATCH --partition=std
#SBATCH --mail-user=kenohagena@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --error=/work/faty014/cluster/slurm_%j.out
#SBATCH --output=/work/faty014/cluster/slurm_%j.err

source /sw/modules/rrz-modules.sh


    '''.format(**{'walltime': walltime,
                  'nodes': nodes,
                  'memory': memory,
                  'tasks': tasks,
                  'name': name,
                  'logdir': logdir})

    environment_variables = '''
module purge
module load env
module load site/hummel
source ~/.bashrc

python3 {script}

cp -r {workdir} /work/faty014
    '''.format(**{'script': script,
                  'workdir': workdir})
    command = sbatch_directives + environment_variables
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='/work/faty014/',
                                     prefix='sbatch_script') as shellfname:
        shellfname.write(command)
        shellfname = shellfname.name
    command = "sbatch %s" % (shellfname)
    output = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT,
        shell=True)
    return output


def to_script(func, tmpdir, *args):
    '''
    Write a simple stub python function that calls this function.
    '''

    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='/work/faty014/',
                                     prefix='py_script') as script:
        code = """
from {module} import {function}
{function}{args}
        """.format(**{'module': func.__module__,
                      'function': func.__name__,
                      'args': args})
        script.write(code)
        return str(script.name)


def pmap(func, *args, walltime=12, memory=10, tmp=None, name=None, tasks=16, env=None, nodes=1):
    if name is None:
        name = func.__name__
    if tmp is None:
        from os.path import join
        tmp = tempfile.TemporaryDirectory().name
    workdir = join(tmp, name)
    tmpdir = join(workdir, 'tmp')
    logdir = join(workdir, 'cluster_logs')
    mkdir_p(tmpdir)
    mkdir_p(logdir)
    out = []
    script = to_script(func, tmpdir, *args)
    print('script in pmap = {}'.format(script))
    pid = submit(walltime=walltime, memory=memory, logdir=logdir,
                 tmpdir=tmpdir, script=script, name=name, nodes=nodes,
                 tasks=tasks, workdir=workdir)
    out.append(pid)
    return out


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
