import os, json
from enum import Enum
import os.path as osp
from collections import namedtuple
import datetime
import dateutil.tz
from .logging import Logger
import errno
import git


def project_root():
    return os.environ.get("PROJECT_ROOT_DIR", os.getcwd())


project_dir = project_root()

GitInfo = namedtuple(
    'GitInfo',
    [
        'directory',
        'code_diff',
        'code_diff_staged',
        'commit_hash',
        'branch_name',
    ],
)

def get_git_infos(dirs):
    git_infos = None
    try:
        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError as e:
                print("Not a valid git repo: {}".format(directory))
    except ImportError:
        git_infos = None
    return git_infos

def create_exp_name(exp_prefix, exp_id=0, seed=None):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    hostname = os.uname()[1].split('.')[0]
    return "%s_%s_%s_%04d_%d" % (exp_prefix, hostname, timestamp, exp_id, 0 if seed is None else seed)


def log_git(log_dir, code_dirs=None):
    if code_dirs is None:
        code_dirs = [project_dir]
    git_infos = get_git_infos(code_dirs)
    if git_infos is not None:
        for (
                directory, code_diff, code_diff_staged, commit_hash, branch_name
        ) in git_infos:
            directory = str(directory)
            if directory[-1] == '/':
                directory = directory[:-1]
            diff_file_name = directory[1:].replace("/", "-") + ".patch"
            diff_staged_file_name = (
                    directory[1:].replace("/", "-") + "_staged.patch"
            )
            if code_diff is not None and len(code_diff) > 0:
                with open(osp.join(log_dir, diff_file_name), "w") as f:
                    f.write(code_diff + '\n')
            if code_diff_staged is not None and len(code_diff_staged) > 0:
                with open(osp.join(log_dir, diff_staged_file_name), "w") as f:
                    f.write(code_diff_staged + '\n')
            with open(osp.join(log_dir, "git_infos.txt"), "a") as f:
                f.write("directory: {}\n".format(directory))
                f.write("git hash: {}\n".format(commit_hash))
                f.write("git branch name: {}\n\n".format(branch_name))


def create_log_dir(
        exp_prefix,
        exp_id=0,
        seed=None,
        base_log_dir=None,
        include_exp_prefix_sub_dir=True,
):
    """
    Creates and returns a unique log directory.
    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id,
                               seed=seed)
    if base_log_dir is None:
        base_log_dir = str(project_dir.joinpath('data'))
    if include_exp_prefix_sub_dir:
        log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    else:
        log_dir = osp.join(base_log_dir, exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d

def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False

def setup_logger(exp_prefix, variant=None, exp_id=0, seed=None, base_log_dir=None, log_dir=None):
    logger = Logger()
    logger.reset()
    log_dir = create_log_dir(exp_prefix, exp_id, seed, base_log_dir)
    log_git(log_dir=log_dir)
    logger.set_snapshot_dir(log_dir)
    tabular_log_path = osp.join(log_dir, 'progress.csv')
    text_log_path = osp.join(log_dir, 'debug.log')
    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)
    logger.log(f"log directory: {log_dir}")
    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, 'variant.json')
        logger.log_variant(variant_log_path, variant)
    return logger

