{
  description = "Python POC development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python312;

        pythonPackages = python.withPackages (ps: with ps; [
          # Add your Python dependencies here
          # ps.requests
          # ps.fastapi
          # ps.uvicorn
          # ps.pandas
          # ps.sqlalchemy
          pip
          virtualenv
        ]);

        # Libraries needed by dynamically linked binaries (like Claude Code)
        nix-ld-libs = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.glib
          pkgs.openssl
          pkgs.libgcc
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonPackages

            # Dev tooling
            pkgs.ruff
            pkgs.pyright

            # Needed for Claude Code CLI
            pkgs.nodejs_22
          ];

          # Provide a dynamic linker + libraries so precompiled binaries
          # (like the Claude Code native binary) can run on NixOS
          NIX_LD = "${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2";
          NIX_LD_LIBRARY_PATH = nix-ld-libs;

          shellHook = ''
            echo "🐍 Python POC environment ready"
            echo "Python: $(python --version)"
            echo "NIX_LD is set — dynamically linked binaries should work"
          '';
        };
      }
    );
}
