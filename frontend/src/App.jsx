import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./components/Login";
import Classifications from "./components/Classifications";
import { Container, CssBaseline } from "@mui/material";

function App() {
  return (
    <Router>
      <CssBaseline />
      <Container maxWidth="lg" style={{ padding: "20px" }}>
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/classifications" element={<Classifications />} />
        </Routes>
      </Container>
    </Router>
  );
}

export default App;
