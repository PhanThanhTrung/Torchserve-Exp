import argparse
import locale
import os
import platform
import re
import subprocess
import sys


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    if output.startswith("├── "):
        return rc, output.strip().replace("├── ", ""), err.strip()
    return rc, output.strip(), err.strip()

def run_and_parse_first_match(command, regex):
    """Returns the first regex match if it exists"""
    rc, out, _ = run(command)
    if rc != 0:
        return "N/A"
    match = re.search(regex, out)
    if match is None:
        return "N/A"
    return match.group(1)


def check_python_version():
    req_version = (3, 8)
    cur_version = sys.version_info

    if not (
        cur_version.major == req_version[0] and cur_version.minor >= req_version[1]
    ):
        print("System version" + str(cur_version))
        print(
            f"TorchServe supports Python {req_version[0]}.{req_version[1]} and higher only. Please upgrade"
        )
        exit(1)


CPP_LINUX_DEPENDENCIES = (
    "autoconf",
    "automake",
    "git",
    "cmake",
    "m4",
    "g++",
    "flex",
    "bison",
    "libgflags-dev",
    "libkrb5-dev",
    "libsasl2-dev",
    "libnuma-dev",
    "pkg-config",
    "libssl-dev",
    "libcap-dev",
    "gperf",
    "libevent-dev",
    "libtool",
    "libboost-all-dev",
    "libjemalloc-dev",
    "libsnappy-dev",
    "wget",
    "unzip",
    "libiberty-dev",
    "liblz4-dev",
    "liblzma-dev",
    "make",
    "zlib1g-dev",
    "binutils-dev",
    "libsodium-dev",
    "libdouble-conversion-dev",
    "ninja-build",
    "clang-tidy",
    "clang-format",
    "build-essential",
    "libgoogle-perftools-dev",
    "rustc",
    "cargo",
    "libunwind-dev",
)

CPP_DARWIN_DEPENDENCIES = (
    "cmake",
    "m4",
    "boost",
    "double-conversion",
    "gperf",
    "libevent",
    "lz4",
    "snappy",
    "xz",
    "openssl",
    "libsodium",
    "icu4c",
    "libomp",
    "llvm",
)

CPP_DARWIN_DEPENDENCIES_LINK = (
    "cmake",
    "boost",
    "double-conversion",
    "gperf",
    "libevent",
    "lz4",
    "snappy",
    "openssl",
    "xz",
    "libsodium",
)


class Common:
    def __init__(self):
        self.torch_stable_url = "https://download.pytorch.org/whl/torch_stable.html"
        self.sudo_cmd = "sudo "

    def install_java(self):
        pass

    def install_nodejs(self):
        pass

    def install_torch_packages(self, cuda_version):
        if cuda_version:
            if platform.system() == "Darwin":
                print(
                    "CUDA not supported on MacOS. Refer https://pytorch.org/ for installing from source."
                )
                sys.exit(1)
            elif cuda_version == "cu92" and platform.system() == "Windows":
                print(
                    "CUDA 9.2 not supported on Windows. Refer https://pytorch.org/ for installing from source."
                )
                sys.exit(1)
            else:
                os.system(
                    f"{sys.executable} -m pip install -U -r requirements/torch_{cuda_version}_{platform.system().lower()}.txt"
                )
        elif args.neuronx:
            torch_neuronx_requirements_file = os.path.join(
                "requirements", "torch_neuronx_linux.txt"
            )
            os.system(
                f"{sys.executable} -m pip install -U -r {torch_neuronx_requirements_file}"
            )
        else:
            if platform.machine() == "aarch64":
                os.system(
                    f"{sys.executable} -m pip install -U -r requirements/torch_{platform.system().lower()}_{platform.machine()}.txt"
                )
            else:
                os.system(
                    f"{sys.executable} -m pip install -U -r requirements/torch_{platform.system().lower()}.txt"
                )

    def install_python_packages(self, cuda_version, requirements_file_path, nightly):
        check = "where" if platform.system() == "Windows" else "which"
        if os.system(f"{check} conda") == 0:
            # conda install command should run before the pip install commands
            # as it may reinstall the packages with different versions
            os.system("conda install -y conda-build")

        # Install PyTorch packages
        if nightly:
            pt_nightly = "cpu" if not cuda_version else cuda_version
            os.system(
                f"pip3 install numpy --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/{pt_nightly}"
            )
        elif args.skip_torch_install:
            print("Skipping Torch installation")
        else:
            self.install_torch_packages(cuda_version)

        # developer.txt also installs packages from common.txt
        os.system(f"{sys.executable} -m pip install -U -r {requirements_file_path}")

        # Install dependencies for GPU
        if not isinstance(cuda_version, type(None)):
            gpu_requirements_file = os.path.join("requirements", "common_gpu.txt")
            os.system(f"{sys.executable} -m pip install -U -r {gpu_requirements_file}")

        # Install dependencies for Inferentia2
        if args.neuronx:
            neuronx_requirements_file = os.path.join("requirements", "neuronx.txt")
            os.system(
                f"{sys.executable} -m pip install -U -r {neuronx_requirements_file}"
            )

    def install_node_packages(self):
        os.system(
            f"{self.sudo_cmd}npm install -g newman@5.3.2 newman-reporter-htmlextra markdown-link-check"
        )

    def install_jmeter(self):
        pass

    def install_wget(self):
        pass

    def install_numactl(self):
        pass

    def install_cpp_dependencies(self):
        raise NotImplementedError(
            f"Cpp backend is not implemented for platform {platform.system()}"
        )

    def install_neuronx_driver(self):
        pass


class Linux(Common):
    def __init__(self):
        super().__init__()
        # Skip 'sudo ' when the user is root
        self.sudo_cmd = "" if os.geteuid() == 0 else self.sudo_cmd

        if args.force:
            os.system(f"{self.sudo_cmd}apt-get update")

    def install_java(self):
        if os.system("javac --version") != 0 or args.force:
            os.system(f"{self.sudo_cmd}apt-get install -y openjdk-17-jdk")

    def install_nodejs(self):
        if os.system("node -v") != 0 or args.force:
            os.system(
                f"{self.sudo_cmd}curl -sL https://deb.nodesource.com/setup_18.x | {self.sudo_cmd}bash -"
            )
            os.system(f"{self.sudo_cmd}apt-get install -y nodejs")

    def install_wget(self):
        if os.system("wget --version") != 0 or args.force:
            os.system(f"{self.sudo_cmd}apt-get install -y wget")

    def install_numactl(self):
        if os.system("numactl --show") != 0 or args.force:
            os.system(f"{self.sudo_cmd}apt-get install -y numactl")

    def install_cpp_dependencies(self):
        os.system(
            f"{self.sudo_cmd}apt-get install -y {' '.join(CPP_LINUX_DEPENDENCIES)}"
        )

    def install_neuronx_driver(self):
        # Configure Linux for Neuron repository updates
        os.system(
            ". /etc/os-release\n"
            + f"{self.sudo_cmd}tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF\n"
            + "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main\n"
            + "EOF\n"
            + "wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -"
        )

        # Update OS packages
        os.system(f"{self.sudo_cmd}apt-get update -y")

        # Install OS headers
        os.system(f"{self.sudo_cmd}apt-get install -y linux-headers-$(uname -r)")

        # install Neuron Driver
        os.system(f"{self.sudo_cmd}apt-get install -y aws-neuronx-dkms")

        # Install Neuron Runtime
        os.system(f"{self.sudo_cmd}apt-get install -y aws-neuronx-collectives")
        os.system(f"{self.sudo_cmd}apt-get install -y aws-neuronx-runtime-lib")

class Darwin(Common):
    def __init__(self):
        super().__init__()

    def install_java(self):
        if os.system("javac -version") != 0 or args.force:
            out = get_brew_version()
            if out == "N/A":
                sys.exit("**Error: Homebrew not installed...")
            os.system("brew install openjdk@17")

    def install_nodejs(self):
        os.system("brew unlink node")
        os.system("brew install node@18")
        os.system("brew link --overwrite node@18")

    def install_node_packages(self):
        os.system(f"{self.sudo_cmd} ./ts_scripts/mac_npm_deps")

    def install_wget(self):
        if os.system("wget --version") != 0 or args.force:
            os.system("brew install wget")

    def install_numactl(self):
        if os.system("numactl --show") != 0 or args.force:
            os.system("brew install numactl")

    def install_cpp_dependencies(self):
        if os.system("clang-tidy --version") != 0 or args.force:
            os.system(f"brew install -f {' '.join(CPP_DARWIN_DEPENDENCIES)}")
            os.system(f"brew link {' '.join(CPP_DARWIN_DEPENDENCIES_LINK)}")
            os.system(
                f'{self.sudo_cmd} ln -s "$(brew --prefix llvm)/bin/clang-format" "/usr/local/bin/clang-format"'
            )
            os.system(
                f'{self.sudo_cmd} ln -s "$(brew --prefix llvm)/bin/clang-tidy" "/usr/local/bin/clang-tidy"'
            )
            os.system(
                f'{self.sudo_cmd} ln -s "$(brew --prefix llvm)/bin/clang-apply-replacements" "/usr/local/bin/clang-apply-replacements"'
            )

    def install_neuronx_driver(self):
        pass


def install_dependencies(cuda_version=None, nightly=False):
    os_map = {"Linux": Linux, "Darwin": Darwin}
    system = os_map[platform.system()]()

    if args.environment == "dev":
        system.install_wget()
        system.install_nodejs()
        system.install_node_packages()
        system.install_numactl()

    # Sequence of installation to be maintained
    system.install_java()

    if args.neuronx:
        system.install_neuronx_driver()

    requirements_file = "common.txt" if args.environment == "prod" else "developer.txt"
    requirements_file_path = os.path.join("requirements", requirements_file)

    system.install_python_packages(cuda_version, requirements_file_path, nightly)

    if args.cpp:
        system.install_cpp_dependencies()


def get_brew_version():
    """Returns `brew --version` output."""

    return run_and_parse_first_match("brew --version", r"Homebrew (.*)")


if __name__ == "__main__":
    check_python_version()
    parser = argparse.ArgumentParser(
        description="Install various build and test dependencies of TorchServe"
    )
    parser.add_argument(
        "--cuda",
        default=None,
        choices=[
            "cu92",
            "cu101",
            "cu102",
            "cu111",
            "cu113",
            "cu116",
            "cu117",
            "cu118",
            "cu121",
        ],
        help="CUDA version for torch",
    )
    parser.add_argument(
        "--neuronx",
        action="store_true",
        help="Install dependencies for inferentia2 support",
    )
    parser.add_argument(
        "--cpp",
        action="store_true",
        help="Install dependencies for cpp backend",
    )
    parser.add_argument(
        "--environment",
        default="prod",
        choices=["prod", "dev"],
        help="environment(production or developer) on which dependencies will be installed",
    )

    parser.add_argument(
        "--nightly_torch",
        action="store_true",
        help="Install nightly version of torch package",
    )

    parser.add_argument(
        "--skip_torch_install",
        action="store_true",
        help="Skip Torch installation",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="force reinstall dependencies wget, node, java and apt-update",
    )
    args = parser.parse_args()

    install_dependencies(cuda_version=args.cuda, nightly=args.nightly_torch)
