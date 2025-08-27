import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Button,
  Alert,
  CircularProgress,
} from "@mui/material";
import api from "../services/api";

const Classifications = () => {
  const [classifications, setClassifications] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [warning, setWarning] = useState(null);

  const fetchClassifications = async () => {
    setLoading(true);
    setError(null);
    setWarning(null);
    try {
      const response = await api.get("/api/results", { timeout: 5000 });
      // Check for duplicates
      const messageIds = response.data.classifications.map(
        (item) => item.message_id
      );
      const duplicates = messageIds.filter(
        (id, index) => messageIds.indexOf(id) !== index
      );
      if (duplicates.length > 0) {
        setWarning(
          `Duplicate message IDs detected: ${duplicates.join(
            ", "
          )}. Showing latest entries.`
        );
      }
      // Remove duplicates, keep latest by timestamp
      const uniqueClassifications = Object.values(
        response.data.classifications.reduce((acc, item) => {
          const existing = acc[item.message_id];
          if (
            !existing ||
            new Date(item.timestamp) > new Date(existing.timestamp)
          ) {
            acc[item.message_id] = item;
          }
          return acc;
        }, {})
      ).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      setClassifications(uniqueClassifications);
    } catch (err) {
      console.error("Fetch classifications error:", err);
      if (err.code === "ERR_NETWORK") {
        setError(
          "Cannot connect to backend. Ensure the server is running at https://localhost:8000."
        );
      } else {
        setError(
          `Error fetching classifications: ${err.message}. Please try again.`
        );
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClassifications();
    const interval = setInterval(fetchClassifications, 30000); // Poll every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <Typography variant="h5" gutterBottom>
        Latest Email Classifications
      </Typography>
      {error && <Alert severity="error">{error}</Alert>}
      {warning && <Alert severity="warning">{warning}</Alert>}
      {loading ? (
        <CircularProgress />
      ) : classifications.length === 0 ? (
        <Alert severity="info">
          No classifications available yet. Authorize Gmail and wait for
          processing.
        </Alert>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Sender</TableCell>
                <TableCell>Subject</TableCell>
                <TableCell>Label</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Timestamp</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {classifications.map((item, index) => (
                <TableRow key={item.message_id || `fallback-${index}`}>
                  <TableCell>{item.sender}</TableCell>
                  <TableCell>{item.subject}</TableCell>
                  <TableCell>{item.label}</TableCell>
                  <TableCell>{item.confidence.toFixed(4)}</TableCell>
                  <TableCell>
                    {new Date(item.timestamp).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      <Button
        variant="contained"
        color="primary"
        onClick={fetchClassifications}
        sx={{ mt: 2 }}
      >
        Refresh
      </Button>
    </div>
  );
};

export default Classifications;
