import React, { useState, useEffect, useMemo } from "react";
import api from "../services/api";
import { Button } from "./ui/button";
import { Alert } from "./ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import { Table as SimpleTable, THead, TBody, TR, TH, TD } from "./ui/table";
import { motion, AnimatePresence } from "framer-motion";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
} from "@tanstack/react-table";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

const Classifications = () => {
  const [classifications, setClassifications] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [warning, setWarning] = useState(null);
  const [globalFilter, setGlobalFilter] = useState("");

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

  const columns = useMemo(
    () => [
      { accessorKey: "sender", header: "Sender" },
      { accessorKey: "subject", header: "Subject" },
      { accessorKey: "label", header: "Label" },
      {
        accessorKey: "confidence",
        header: "Confidence",
        cell: (info) => info.getValue().toFixed(4),
      },
      {
        accessorKey: "timestamp",
        header: "Timestamp",
        cell: (info) => new Date(info.getValue()).toLocaleString(),
      },
    ],
    []
  );

  const table = useReactTable({
    data: classifications,
    columns,
    state: { globalFilter },
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  const stats = useMemo(() => {
    const total = classifications.length;
    const spam = classifications.filter(
      (c) => (c.label || "").toLowerCase() === "spam"
    ).length;
    const ham = classifications.filter(
      (c) => (c.label || "").toLowerCase() !== "spam"
    ).length;
    const avgConfidence = classifications.length
      ? classifications.reduce(
          (acc, c) => acc + (Number(c.confidence) || 0),
          0
        ) / classifications.length
      : 0;
    return { total, spam, ham, avgConfidence };
  }, [classifications]);

  const chartData = useMemo(() => {
    const byDay = new Map();
    for (const c of classifications) {
      const day = new Date(c.timestamp).toLocaleDateString();
      const entry = byDay.get(day) || { day, spam: 0, ham: 0 };
      if ((c.label || "").toLowerCase() === "spam") entry.spam += 1;
      else entry.ham += 1;
      byDay.set(day, entry);
    }
    return Array.from(byDay.values()).sort(
      (a, b) => new Date(a.day) - new Date(b.day)
    );
  }, [classifications]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card>
        <CardHeader>
          <CardTitle>Latest Email Classifications</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && <Alert variant="error">{error}</Alert>}
          {warning && <Alert variant="warning">{warning}</Alert>}

          {loading ? (
            <div className="flex items-center gap-2 text-sm text-[hsl(var(--muted-foreground))]">
              <Spinner /> Loading...
            </div>
          ) : classifications.length === 0 ? (
            <Alert variant="info">
              No classifications yet. Authorize Gmail and wait for processing.
            </Alert>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Card className="glass p-3">
                  <div className="text-xs text-[hsl(var(--muted-foreground))]">
                    Total
                  </div>
                  <div className="text-2xl font-semibold">{stats.total}</div>
                </Card>
                <Card className="glass p-3">
                  <div className="text-xs text-[hsl(var(--muted-foreground))]">
                    Spam
                  </div>
                  <div className="text-2xl font-semibold text-red-600 dark:text-red-400">
                    {stats.spam}
                  </div>
                </Card>
                <Card className="glass p-3">
                  <div className="text-xs text-[hsl(var(--muted-foreground))]">
                    Ham
                  </div>
                  <div className="text-2xl font-semibold text-emerald-600 dark:text-emerald-400">
                    {stats.ham}
                  </div>
                </Card>
                <Card className="glass p-3">
                  <div className="text-xs text-[hsl(var(--muted-foreground))]">
                    Avg Confidence
                  </div>
                  <div className="text-2xl font-semibold">
                    {stats.avgConfidence.toFixed(2)}
                  </div>
                </Card>
              </div>
              <div className="h-64 w-full rounded-md border border-[hsl(var(--border))] p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <XAxis
                      dataKey="day"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      allowDecimals={false}
                    />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="spam" stackId="a" fill="#ef4444" />
                    <Bar dataKey="ham" stackId="a" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="flex items-center justify-between gap-2">
                <input
                  className="h-9 w-full max-w-xs rounded-md border border-[hsl(var(--border))] bg-transparent px-3 text-sm outline-none focus:ring-2 focus:ring-[hsl(var(--ring))]"
                  placeholder="Search..."
                  value={globalFilter ?? ""}
                  onChange={(e) => setGlobalFilter(e.target.value)}
                />
                <div className="flex items-center gap-2">
                  <Button
                    onClick={() => table.setPageIndex(0)}
                    disabled={!table.getCanPreviousPage()}
                  >
                    First
                  </Button>
                  <Button
                    onClick={() => table.previousPage()}
                    disabled={!table.getCanPreviousPage()}
                  >
                    Prev
                  </Button>
                  <Button
                    onClick={() => table.nextPage()}
                    disabled={!table.getCanNextPage()}
                  >
                    Next
                  </Button>
                </div>
              </div>
              <SimpleTable>
                <THead>
                  {table.getHeaderGroups().map((headerGroup) => (
                    <TR key={headerGroup.id}>
                      {headerGroup.headers.map((header) => (
                        <TH
                          key={header.id}
                          onClick={header.column.getToggleSortingHandler()}
                          className="cursor-pointer select-none"
                        >
                          {flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                        </TH>
                      ))}
                    </TR>
                  ))}
                </THead>
                <TBody>
                  <AnimatePresence initial={false}>
                    {table.getRowModel().rows.map((row) => (
                      <motion.tr
                        key={row.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                      >
                        {row.getVisibleCells().map((cell) => (
                          <TD key={cell.id}>
                            {flexRender(
                              cell.column.columnDef.cell,
                              cell.getContext()
                            )}
                          </TD>
                        ))}
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </TBody>
              </SimpleTable>
            </div>
          )}

          <div className="pt-2">
            <Button variant="secondary" onClick={fetchClassifications}>
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default Classifications;
