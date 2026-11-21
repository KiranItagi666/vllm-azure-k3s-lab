
        param([string]$RepoRoot = (Split-Path -Parent $PSScriptRoot))

        $path = Join-Path $RepoRoot 'diagnostics/vllm-diagnostics-milestone-2.json'
        $manifest = Get-Content -Raw -LiteralPath $path -ErrorAction Stop |
            ConvertFrom-Json -ErrorAction Stop

        $namespace = 'vllm-example'
        $identity = 'vllm-diagnostics'

        function Get-LabResource($kind, $name) {
            $matches = @($manifest.items | Where-Object {
                $_.kind -ceq $kind -and
                $_.metadata.name -ceq $name -and
                $_.metadata.namespace -ceq $namespace
            })
            if ($matches.Count -ne 1) {
                throw "Expected one $kind named $name in $namespace."
            }
            $matches[0]
        }

        $account = Get-LabResource 'ServiceAccount' $identity
        $role = Get-LabResource 'Role' $identity
        $binding = Get-LabResource 'RoleBinding' $identity
        $deployment = Get-LabResource 'Deployment' 'vllm-diagnostics-mcp'
        $subjects = @($binding.subjects)

        if ($deployment.spec.template.spec.serviceAccountName -cne $account.metadata.name -or
            $binding.roleRef.kind -cne 'Role' -or
            $binding.roleRef.name -cne $role.metadata.name -or
            $subjects.Count -ne 1 -or
            $subjects[0].kind -cne 'ServiceAccount' -or
            $subjects[0].name -cne $account.metadata.name -or
            $subjects[0].namespace -cne $namespace) {
            throw 'The diagnostic service account wiring differs from the verified setup.'
        }

        $actual = foreach ($rule in $role.rules) {
            if ($rule.resourceNames -or $rule.nonResourceURLs) {
                throw 'The diagnostic role contains an unexpected rule.'
            }
            foreach ($group in $rule.apiGroups) {
                foreach ($resource in $rule.resources) {
                    foreach ($verb in $rule.verbs) {
                        "$group|$resource|$verb"
                    }
                }
            }
        }

        $expected = '|events|list', '|pods|list', 'apps|deployments|list', 'apps|replicasets|list'
        if ((($actual | Sort-Object -Unique) -join "`n") -cne (($expected | Sort-Object) -join "`n")) {
            throw 'The diagnostic role differs from the four verified list permissions.'
        }

        Write-Output 'PASS - Diagnostic deployment and read-only permissions verified.'
    
