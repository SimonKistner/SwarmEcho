"""GPU attribution checks; standard-library unittest, no live cluster required."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"cluster/tools"))
from capacity import collect, summarize, task_metadata


def job(identity, slots, kind="TYPE_EXPERIMENT", workspace=129):
    return dict(jobId=identity, allocatedSlots=slots, requestedSlots=slots+7,
        type=kind, workspaceId=workspace)


def meta(identity, project=367, user="me", kind="experiment"):
    return dict(id=identity, name=identity, project_id=project, project="swarm_echo",
        username=user, kind=kind, state="RUNNING")


class CapacityTests(unittest.TestCase):
    def test_slot_counts_overlap_and_use_ids_not_names(self):
        jobs=[job("own",3),job("same-project-other",4),job("other-project-own",2),
            job("other-project-other",1),job("shell-other",3,"TYPE_SHELL"),
            job("shell-own",1,"TYPE_SHELL"),job("queued",0),job("elsewhere",20,workspace=999)]
        metadata={"own":meta("own"),"same-project-other":meta("same-project-other",user="other"),
            "other-project-own":meta("other-project-own",project=555),
            "other-project-other":meta("other-project-other",project=555,user="other"),
            "shell-other":meta("shell-other",project=None,user="other",kind="shell"),
            "shell-own":meta("shell-own",project=None,kind="shell")}
        result=summarize(jobs,metadata,workspace_id=129,project_id=367,username="me")
        self.assertEqual(result["allocated_slots"],14)
        self.assertEqual(result["outside_project_slots"],7)
        self.assertEqual(result["outside_user_slots"],8)
        self.assertEqual(result["free_slots"],10)
        self.assertEqual(len(result["allocations"]),6)

    def test_missing_metadata_is_reported_unknown(self):
        result=summarize([job("gone",2)],{},workspace_id=129,project_id=367,username="me")
        self.assertEqual(result["allocated_slots"],2)
        self.assertEqual(result["unknown_project_slots"],2)
        self.assertEqual(result["unknown_user_slots"],2)
        self.assertEqual(result["outside_project_slots"],0)
        self.assertEqual(result["outside_user_slots"],0)

    def test_zero_gpu_shells_do_not_consume_slots_or_ownership_counts(self):
        jobs=[job("gpu",3,"TYPE_SHELL"),job("zero",0,"TYPE_SHELL"),job("waiting",0,"TYPE_SHELL")]
        metadata={"gpu":meta("gpu",project=None,user="other",kind="shell"),
            "zero":meta("zero",project=None,user="other",kind="shell"),
            "waiting":dict(meta("waiting",project=None,kind="shell"),state="PENDING")}
        result=summarize(jobs,metadata,workspace_id=129,project_id=367,username="me")
        self.assertEqual(result["allocated_slots"],3)
        self.assertEqual(result["running_shells"],2)
        self.assertEqual(result["zero_gpu_shells"],1)
        self.assertEqual(result["outside_project_slots"],3)
        self.assertEqual(result["outside_user_slots"],3)

    def test_task_metadata_drops_connection_keys(self):
        metadata=task_metadata([dict(jobId="shell",id="uuid",username="other",description="arc",
            state="RUNNING",privateKey="private",publicKey="public",addresses=["connection"])],"shell")
        encoded=json.dumps(metadata)
        self.assertNotIn("private",encoded)
        self.assertNotIn("public",encoded)
        self.assertNotIn("connection",encoded)
        self.assertIsNone(metadata["shell"]["project_id"])

    def test_collection_joins_authoritative_allocations_with_metadata(self):
        class FakeDet:
            def text(self,*args):
                if args[:2]==("user","whoami"):
                    return "You are logged in as user 'me'"
                raise AssertionError("Historical experiment listing should not be queried.")
            def json(self,*args,**kwargs):
                if args[:2]==("project","describe"):
                    return dict(id=367,workspaceId=129)
                if args[:2]==("job","list"):
                    return dict(jobs=[job("own",1),job("other",2),job("shell",3,"TYPE_SHELL")])
                if args[:2]==("task","list"):
                    return {"a":dict(taskId="10.uuid",resourcePool="42_Compute"),
                            "b":dict(taskId="11.uuid",resourcePool="42_Compute")}
                if args[:2]==("experiment","describe"):
                    return [dict(experiment=dict(id=10,jobId="own",name="own",username="me",projectId=367)),
                            dict(experiment=dict(id=11,jobId="other",name="other",username="other",projectId=555))]
                return [dict(id="shell-id",jobId="shell",username="other",description="arc",state="RUNNING")]
        result=collect(dict(workspace="RAI_WS",project="swarm_echo",project_id=367,resource_pool="42_Compute"),FakeDet())
        self.assertEqual(result["allocated_slots"],6)
        self.assertEqual(result["outside_project_slots"],5)
        self.assertEqual(result["outside_user_slots"],5)
        self.assertEqual(result["unknown_project_slots"],0)
        self.assertEqual(result["username"],"me")


if __name__=="__main__":
    unittest.main()
