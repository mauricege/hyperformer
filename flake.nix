{
  description = "A simple flake for hyperformer";

  inputs.nixpkgs.url = "nixpkgs/release-22.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.pypi-deps-db = {
    flake = false;
    url = "github:DavHau/pypi-deps-db";
  };
  inputs.mach-nix = {
    url = "mach-nix/3.5.0";
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.pypi-deps-db.follows = "pypi-deps-db";
  };

  outputs = { self, nixpkgs, mach-nix, flake-utils, pypi-deps-db, ... }:
    let
      name = "hyperformer";
      commonOptions = {
        ignoreDataOutdated = true;
        python = "python39";
        src = ./.;
        providers.torch = "nixpkgs";
        providers.torchaudio = "nixpkgs";
        providers.black = "nixpkgs";
        overridesPost = [(final: prev: {
          torch = prev.pytorch-bin;
          torchaudio = prev.torchaudio-bin;
        })];
      };
      developmentEnv = { pkgs }: mach-nix.lib.${pkgs.system}.buildPythonPackage {
        inherit (commonOptions) ignoreDataOutdated python src providers overridesPost;
        pname = name;
        version = "0.0.1";
        requirements = builtins.readFile ./requirements.txt;
      };
      shell = { pkgs }: pkgs.mkShell {
        name = "${name}-dev";
        shellHook = ''
          unset SOURCE_DATE_EPOCH
        '';
        buildInputs = [
          (developmentEnv { inherit pkgs; })
        ];
      };
    in
    flake-utils.lib.simpleFlake
      {
        inherit self nixpkgs shell name;
        overlay = final: prev: {
          hyperformer = {
            defaultPackage = developmentEnv { pkgs = prev; };
            hyperformer = developmentEnv { pkgs = prev; };
          };
        };
        config = { allowUnfree = true; };
        systems = [ "x86_64-linux" ];
      };
}
