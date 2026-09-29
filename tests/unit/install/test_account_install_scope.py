from unittest.mock import create_autospec

import pytest
from databricks.labs.blueprint.tui import MockPrompts, Prompts
from databricks.labs.blueprint.wheels import ProductInfo
from databricks.sdk import AccountClient, WorkspaceClient
from databricks.sdk.service import iam
from databricks.sdk.service.iam import ComplexValue
from databricks.sdk.service.provisioning import Workspace

from databricks.labs.ucx.config import WorkspaceConfig
from databricks.labs.ucx.install import AccountInstaller


def _account_with_admin_status(admin_by_id: dict[int, bool]):
    acc = create_autospec(AccountClient)
    workspaces = []
    clients = {}
    for workspace_id, is_admin in admin_by_id.items():
        workspaces.append(Workspace(workspace_id=workspace_id, deployment_name=f"ws-{workspace_id}"))
        client = create_autospec(WorkspaceClient)
        groups = [ComplexValue(display="admins")] if is_admin else [ComplexValue(display="users")]
        client.current_user.me.return_value = iam.User(user_name="me@example.com", groups=groups)
        clients[workspace_id] = client
    acc.workspaces.list.return_value = workspaces
    acc.get_workspace_client.side_effect = lambda w: clients[w.workspace_id]
    return acc, clients


def test_account_installer_fails_when_requested_workspace_id_not_in_account():
    acc, clients = _account_with_admin_status({123: True})
    account_installer = AccountInstaller(acc, {"workspace_ids": "123,999"}).replace(
        prompts=MockPrompts({r".*": "Yes"}),
        product_info=ProductInfo.for_testing(WorkspaceConfig),
        environ={},
    )

    with pytest.raises(SystemExit, match="999: not found in account"):
        account_installer.install_on_account()
    clients[123].workspace.upload.assert_not_called()


def test_account_installer_fails_when_requested_workspace_id_not_administrable():
    acc, clients = _account_with_admin_status({123: True, 456: False})
    account_installer = AccountInstaller(acc, {"workspace_ids": "123,456"}).replace(
        prompts=MockPrompts({r".*": "Yes"}),
        product_info=ProductInfo.for_testing(WorkspaceConfig),
        environ={},
    )

    with pytest.raises(SystemExit, match="456: not a workspace admin or no access"):
        account_installer.install_on_account()
    clients[123].workspace.upload.assert_not_called()


def test_account_installer_without_requested_ids_still_skips_non_admin_workspaces():
    """Upstream behavior is unchanged when no workspace ids are requested: non-admin workspaces are skipped, not fatal."""
    acc, _ = _account_with_admin_status({123: True, 456: False})
    prompts = create_autospec(Prompts)
    prompts.confirm.return_value = False
    account_installer = AccountInstaller(acc).replace(
        prompts=prompts,
        product_info=ProductInfo.for_testing(WorkspaceConfig),
        environ={},
    )

    account_installer.install_on_account()

    confirmation = prompts.confirm.call_args.args[0]
    assert "ws-123" in confirmation
    assert "ws-456" not in confirmation
