import logging
from unittest import mock

import pytest
from databricks.labs.blueprint.installation import MockInstallation
from databricks.labs.blueprint.installer import InstallState
from databricks.labs.blueprint.wheels import ProductInfo, WheelsV2, find_project_root
from databricks.labs.lsql.backends import MockBackend
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied

from databricks.labs.ucx.config import WorkspaceConfig
from databricks.labs.ucx.install import INSTALL_SCOPES, WorkspaceInstallation, WorkspaceInstaller, load_install_scope
from databricks.labs.ucx.installer.workflows import DeployedWorkflows, WorkflowNotDeployed, WorkflowsDeployment
from databricks.labs.ucx.runtime import Workflows

PRODUCT_INFO = ProductInfo.from_class(WorkspaceConfig)
QUERIES_FOLDER = find_project_root(__file__) / "src/databricks/labs/ucx/queries"


def _workspace_installation(ws, prompts, installation, dashboards):
    install_state = InstallState.from_installation(installation)
    workflows_installation = WorkflowsDeployment(
        WorkspaceConfig(inventory_database="..."),
        installation,
        install_state,
        ws,
        mock.create_autospec(WheelsV2),
        PRODUCT_INFO,
        Workflows([]),
    )
    workspace_installation = WorkspaceInstallation(
        WorkspaceConfig(inventory_database="ucx"),
        installation,
        install_state,
        MockBackend(),
        ws,
        workflows_installation,
        prompts,
        PRODUCT_INFO,
        dashboards=dashboards,
    )
    return workspace_installation, install_state


def test_load_install_scope_unset_deploys_everything() -> None:
    assert load_install_scope({}) is None
    assert load_install_scope({"UCX_INSTALL_SCOPE": " "}) is None


def test_load_install_scope_is_case_insensitive() -> None:
    assert load_install_scope({"UCX_INSTALL_SCOPE": "Assessment"}) == INSTALL_SCOPES["assessment"]


def test_load_install_scope_unknown_value_fails() -> None:
    with pytest.raises(SystemExit, match="unknown scope 'migration', expected one of: assessment"):
        load_install_scope({"UCX_INSTALL_SCOPE": "migration"})


def test_assessment_scope_references_existing_workflows_and_dashboards() -> None:
    """Guard against upstream renaming a workflow or dashboard folder, which would silently empty the scope."""
    scope = INSTALL_SCOPES["assessment"]
    assert scope.workflows <= set(Workflows.all().workflows)
    for dashboard in scope.dashboards:
        assert (QUERIES_FOLDER / dashboard).is_dir(), dashboard


def test_workspace_installer_scopes_workflows() -> None:
    installer = WorkspaceInstaller(mock.create_autospec(WorkspaceClient), {"UCX_INSTALL_SCOPE": "assessment"})
    # pylint: disable-next=protected-access
    assert set(installer._workflows.workflows) == {"assessment", "assess-workflows"}


def test_workspace_installer_without_scope_keeps_all_workflows() -> None:
    installer = WorkspaceInstaller(mock.create_autospec(WorkspaceClient), {"UCX_FORCE_INSTALL": "user"})
    # pylint: disable-next=protected-access
    assert set(installer._workflows.workflows) == set(Workflows.all().workflows)


def test_scoped_install_creates_only_scoped_dashboards(ws, any_prompt) -> None:
    dashboards = INSTALL_SCOPES["assessment"].dashboards
    workspace_installation, install_state = _workspace_installation(ws, any_prompt, MockInstallation(), dashboards)

    workspace_installation.run()

    assert set(install_state.dashboards) == {
        "assessment_main",
        "assessment_estimates",
        "assessment_interactive",
        "assessment_azure",
    }
    assert ws.lakeview.create.call_count == len(dashboards)


def test_unscoped_install_creates_all_dashboards(ws, any_prompt) -> None:
    workspace_installation, install_state = _workspace_installation(ws, any_prompt, MockInstallation(), None)

    workspace_installation.run()

    expected = {f"{d.parent.name}_{d.name}".lower() for d in QUERIES_FOLDER.glob("*/*") if d.is_dir()}
    assert set(install_state.dashboards) == expected
    ws.lakeview.trash.assert_not_called()


def test_scoped_reinstall_trashes_dashboards_outside_scope(ws, any_prompt) -> None:
    installation = MockInstallation(
        {"state.json": {"resources": {"dashboards": {"migration_main": "01ef0mig", "progress_main": "01ef0prog"}}}}
    )
    ws.lakeview.trash.side_effect = [None, NotFound("already gone")]
    workspace_installation, install_state = _workspace_installation(
        ws, any_prompt, installation, INSTALL_SCOPES["assessment"].dashboards
    )

    workspace_installation.run()

    trashed = {call.args[0] for call in ws.lakeview.trash.call_args_list}
    assert trashed == {"01ef0mig", "01ef0prog"}
    assert "migration_main" not in install_state.dashboards
    assert "progress_main" not in install_state.dashboards


def test_scoped_reinstall_deletes_legacy_redash_dashboard_outside_scope(ws, any_prompt) -> None:
    """Redash ids contain '-' and must go through the Redash API, as in _upgrade_redash_dashboard."""
    installation = MockInstallation({"state.json": {"resources": {"dashboards": {"migration_main": "redash-id"}}}})
    workspace_installation, install_state = _workspace_installation(
        ws, any_prompt, installation, INSTALL_SCOPES["assessment"].dashboards
    )

    workspace_installation.run()

    ws.dashboards.delete.assert_called_once_with(dashboard_id="redash-id")
    ws.lakeview.trash.assert_not_called()
    assert "migration_main" not in install_state.dashboards


def test_scoped_reinstall_does_not_fail_when_dashboard_cannot_be_removed(ws, any_prompt, caplog) -> None:
    installation = MockInstallation({"state.json": {"resources": {"dashboards": {"migration_main": "mig"}}}})
    ws.lakeview.trash.side_effect = PermissionDenied("not owner")
    workspace_installation, install_state = _workspace_installation(
        ws, any_prompt, installation, INSTALL_SCOPES["assessment"].dashboards
    )

    with caplog.at_level(logging.WARNING, logger="databricks.labs.ucx.install"):
        assert workspace_installation.run()

    assert "Cannot remove dashboard migration_main (mig), remove it manually" in caplog.text
    assert "migration_main" not in install_state.dashboards


def test_out_of_scope_dashboards_removed_after_jobs_and_readme() -> None:
    """create_jobs() renders the README from install_state.dashboards concurrently with dashboard creation, so
    removal must not mutate that dict until both install tasks have finished."""
    # pylint: disable=protected-access
    workspace_installation = mock.MagicMock()
    calls: list[str] = []
    workspace_installation._create_database_and_dashboards.side_effect = lambda: calls.append("dashboards")
    workspace_installation._workflows_installer.create_jobs.side_effect = lambda: calls.append("jobs")
    workspace_installation._remove_dashboards_outside_scope.side_effect = lambda: calls.append("remove")
    workspace_installation.config.trigger_job = False
    workspace_installation._is_account_install = True

    WorkspaceInstallation.run(workspace_installation)

    assert calls[-1] == "remove"
    assert set(calls[:2]) == {"dashboards", "jobs"}


def test_running_workflow_outside_scope_fails_with_clear_error() -> None:
    ws = mock.create_autospec(WorkspaceClient)
    installation = MockInstallation({"state.json": {"resources": {"jobs": {"assessment": "123"}}}})
    deployed = DeployedWorkflows(ws, InstallState.from_installation(installation))

    with pytest.raises(WorkflowNotDeployed, match="'migrate-tables' is not deployed.*UCX_INSTALL_SCOPE"):
        deployed.run_workflow("migrate-tables")
    ws.jobs.run_now.assert_not_called()


def test_workflow_not_deployed_is_a_key_error() -> None:
    """Callers that caught KeyError from the previous `install_state.jobs[step]` lookup keep working."""
    assert issubclass(WorkflowNotDeployed, KeyError)
