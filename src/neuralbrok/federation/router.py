"""
Federation Router: Handles PII redaction and cross-node communication.
"""
import logging
from typing import Dict, Any

from neuralbrok.federation.crypto import FederationCrypto
from neuralbrok.security.pii_redactor import PIIRedactor
from neuralbrok.security.injection_shield import InjectionShield

logger = logging.getLogger(__name__)

class FederationRouter:
    def __init__(self):
        self.crypto = FederationCrypto()
        self.redactor = PIIRedactor()
        self.shield = InjectionShield()
        self.trusted_peers = {}  # node_id -> trust_score

    def prepare_outbound(self, target_node: str, task: str) -> Dict[str, Any]:
        """Redacts PII, signs the payload, and prepares it for federation transmission."""
        sanitized_task = self.redactor.sanitize(task)
        
        payload = {
            "type": "task_request",
            "target": target_node,
            "content": sanitized_task
        }
        
        return self.crypto.sign_payload(payload)

    def process_inbound(self, signed_message: Dict[str, Any]) -> Dict[str, Any]:
        """Verifies signature, checks safety, and routes to local SwarmCoordinator."""
        if not self.crypto.verify_payload(signed_message):
            logger.warning("Rejected federated message: Invalid signature.")
            return {"error": "Invalid signature"}
            
        payload = signed_message.get("payload", {})
        content = payload.get("content", "")
        sender_id = signed_message.get("node_id", "unknown")
        
        if not self.shield.is_safe(content):
            logger.warning(f"Rejected federated message from {sender_id}: Injection detected.")
            # Downgrade trust
            self.trusted_peers[sender_id] = self.trusted_peers.get(sender_id, 1.0) - 0.5
            return {"error": "Security policy violation"}
            
        # Upgrade trust on successful safe delivery
        self.trusted_peers[sender_id] = min(1.0, self.trusted_peers.get(sender_id, 0.5) + 0.1)
        
        logger.info(f"Accepted valid federated task from node {sender_id}")
        return {"status": "accepted", "task": content}
