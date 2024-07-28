/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"




PyDoc_STRVAR(doc_FbxLimits_GetActive, "GetActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetActive, doc_FbxLimits_GetActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetActive, "SetActive(self, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0))
        {
            sipCpp->SetActive(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetActive, doc_FbxLimits_SetActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMinXActive, "GetMinXActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetMinXActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMinXActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetMinXActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMinXActive, doc_FbxLimits_GetMinXActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMinYActive, "GetMinYActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetMinYActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMinYActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetMinYActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMinYActive, doc_FbxLimits_GetMinYActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMinZActive, "GetMinZActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetMinZActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMinZActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetMinZActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMinZActive, doc_FbxLimits_GetMinZActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMinActive, "GetMinActive(self) -> Tuple[bool, bool, bool]");

extern "C" {static PyObject *meth_FbxLimits_GetMinActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMinActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
        bool a1;
        bool a2;
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            sipCpp->GetMinActive(a0,a1,a2);

            return sipBuildResult(0,"(bbb)",a0,a1,a2);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMinActive, doc_FbxLimits_GetMinActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMin, "GetMin(self) -> FbxDouble3");

extern "C" {static PyObject *meth_FbxLimits_GetMin(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMin(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
             ::FbxDouble3*sipRes;

            sipRes = new  ::FbxDouble3(sipCpp->GetMin());

            return sipConvertFromNewType(sipRes,sipType_FbxDouble3,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMin, doc_FbxLimits_GetMin);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMinXActive, "SetMinXActive(self, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMinXActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMinXActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0))
        {
            sipCpp->SetMinXActive(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMinXActive, doc_FbxLimits_SetMinXActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMinYActive, "SetMinYActive(self, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMinYActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMinYActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0))
        {
            sipCpp->SetMinYActive(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMinYActive, doc_FbxLimits_SetMinYActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMinZActive, "SetMinZActive(self, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMinZActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMinZActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0))
        {
            sipCpp->SetMinZActive(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMinZActive, doc_FbxLimits_SetMinZActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMinActive, "SetMinActive(self, bool, bool, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMinActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMinActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
        bool a1;
        bool a2;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bbbb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0, &a1, &a2))
        {
            sipCpp->SetMinActive(a0,a1,a2);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMinActive, doc_FbxLimits_SetMinActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMin, "SetMin(self, FbxDouble3)");

extern "C" {static PyObject *meth_FbxLimits_SetMin(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMin(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxDouble3* a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLimits, &sipCpp, sipType_FbxDouble3, &a0))
        {
            sipCpp->SetMin(*a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMin, doc_FbxLimits_SetMin);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMaxXActive, "GetMaxXActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetMaxXActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMaxXActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetMaxXActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMaxXActive, doc_FbxLimits_GetMaxXActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMaxYActive, "GetMaxYActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetMaxYActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMaxYActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetMaxYActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMaxYActive, doc_FbxLimits_GetMaxYActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMaxZActive, "GetMaxZActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetMaxZActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMaxZActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetMaxZActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMaxZActive, doc_FbxLimits_GetMaxZActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMaxActive, "GetMaxActive(self) -> Tuple[bool, bool, bool]");

extern "C" {static PyObject *meth_FbxLimits_GetMaxActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMaxActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
        bool a1;
        bool a2;
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            sipCpp->GetMaxActive(a0,a1,a2);

            return sipBuildResult(0,"(bbb)",a0,a1,a2);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMaxActive, doc_FbxLimits_GetMaxActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetMax, "GetMax(self) -> FbxDouble3");

extern "C" {static PyObject *meth_FbxLimits_GetMax(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetMax(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
             ::FbxDouble3*sipRes;

            sipRes = new  ::FbxDouble3(sipCpp->GetMax());

            return sipConvertFromNewType(sipRes,sipType_FbxDouble3,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetMax, doc_FbxLimits_GetMax);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMaxXActive, "SetMaxXActive(self, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMaxXActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMaxXActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0))
        {
            sipCpp->SetMaxXActive(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMaxXActive, doc_FbxLimits_SetMaxXActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMaxYActive, "SetMaxYActive(self, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMaxYActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMaxYActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0))
        {
            sipCpp->SetMaxYActive(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMaxYActive, doc_FbxLimits_SetMaxYActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMaxZActive, "SetMaxZActive(self, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMaxZActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMaxZActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0))
        {
            sipCpp->SetMaxZActive(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMaxZActive, doc_FbxLimits_SetMaxZActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMaxActive, "SetMaxActive(self, bool, bool, bool)");

extern "C" {static PyObject *meth_FbxLimits_SetMaxActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMaxActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0;
        bool a1;
        bool a2;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bbbb", &sipSelf, sipType_FbxLimits, &sipCpp, &a0, &a1, &a2))
        {
            sipCpp->SetMaxActive(a0,a1,a2);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMaxActive, doc_FbxLimits_SetMaxActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_SetMax, "SetMax(self, FbxDouble3)");

extern "C" {static PyObject *meth_FbxLimits_SetMax(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_SetMax(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxDouble3* a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLimits, &sipCpp, sipType_FbxDouble3, &a0))
        {
            sipCpp->SetMax(*a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_SetMax, doc_FbxLimits_SetMax);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_GetAnyMinMaxActive, "GetAnyMinMaxActive(self) -> bool");

extern "C" {static PyObject *meth_FbxLimits_GetAnyMinMaxActive(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_GetAnyMinMaxActive(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLimits, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->GetAnyMinMaxActive();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_GetAnyMinMaxActive, doc_FbxLimits_GetAnyMinMaxActive);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLimits_Apply, "Apply(self, FbxDouble3) -> FbxDouble3");

extern "C" {static PyObject *meth_FbxLimits_Apply(PyObject *, PyObject *);}
static PyObject *meth_FbxLimits_Apply(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxDouble3* a0;
         ::FbxLimits *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLimits, &sipCpp, sipType_FbxDouble3, &a0))
        {
             ::FbxDouble3*sipRes;

            sipRes = new  ::FbxDouble3(sipCpp->Apply(*a0));

            return sipConvertFromNewType(sipRes,sipType_FbxDouble3,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLimits, sipName_Apply, doc_FbxLimits_Apply);

    return NULL;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxLimits(void *, int);}
static void release_FbxLimits(void *sipCppV, int)
{
    delete reinterpret_cast< ::FbxLimits *>(sipCppV);
}


extern "C" {static void assign_FbxLimits(void *, SIP_SSIZE_T, const void *);}
static void assign_FbxLimits(void *sipDst, SIP_SSIZE_T sipDstIdx, const void *sipSrc)
{
    reinterpret_cast< ::FbxLimits *>(sipDst)[sipDstIdx] = *reinterpret_cast<const  ::FbxLimits *>(sipSrc);
}


extern "C" {static void *array_FbxLimits(SIP_SSIZE_T);}
static void *array_FbxLimits(SIP_SSIZE_T sipNrElem)
{
    return new  ::FbxLimits[sipNrElem];
}


extern "C" {static void *copy_FbxLimits(const void *, SIP_SSIZE_T);}
static void *copy_FbxLimits(const void *sipSrc, SIP_SSIZE_T sipSrcIdx)
{
    return new  ::FbxLimits(reinterpret_cast<const  ::FbxLimits *>(sipSrc)[sipSrcIdx]);
}


extern "C" {static void dealloc_FbxLimits(sipSimpleWrapper *);}
static void dealloc_FbxLimits(sipSimpleWrapper *sipSelf)
{
    if (sipIsOwnedByPython(sipSelf))
    {
        release_FbxLimits(sipGetAddress(sipSelf), 0);
    }
}


extern "C" {static void *init_type_FbxLimits(sipSimpleWrapper *, PyObject *, PyObject *, PyObject **, PyObject **, PyObject **);}
static void *init_type_FbxLimits(sipSimpleWrapper *, PyObject *sipArgs, PyObject *sipKwds, PyObject **sipUnused, PyObject **, PyObject **sipParseErr)
{
     ::FbxLimits *sipCpp = 0;

    {
        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, ""))
        {
            sipCpp = new  ::FbxLimits();

            return sipCpp;
        }
    }

    {
        const  ::FbxLimits* a0;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "J9", sipType_FbxLimits, &a0))
        {
            sipCpp = new  ::FbxLimits(*a0);

            return sipCpp;
        }
    }

    return NULL;
}


static PyMethodDef methods_FbxLimits[] = {
    {SIP_MLNAME_CAST(sipName_Apply), meth_FbxLimits_Apply, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_Apply)},
    {SIP_MLNAME_CAST(sipName_GetActive), meth_FbxLimits_GetActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetActive)},
    {SIP_MLNAME_CAST(sipName_GetAnyMinMaxActive), meth_FbxLimits_GetAnyMinMaxActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetAnyMinMaxActive)},
    {SIP_MLNAME_CAST(sipName_GetMax), meth_FbxLimits_GetMax, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMax)},
    {SIP_MLNAME_CAST(sipName_GetMaxActive), meth_FbxLimits_GetMaxActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMaxActive)},
    {SIP_MLNAME_CAST(sipName_GetMaxXActive), meth_FbxLimits_GetMaxXActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMaxXActive)},
    {SIP_MLNAME_CAST(sipName_GetMaxYActive), meth_FbxLimits_GetMaxYActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMaxYActive)},
    {SIP_MLNAME_CAST(sipName_GetMaxZActive), meth_FbxLimits_GetMaxZActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMaxZActive)},
    {SIP_MLNAME_CAST(sipName_GetMin), meth_FbxLimits_GetMin, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMin)},
    {SIP_MLNAME_CAST(sipName_GetMinActive), meth_FbxLimits_GetMinActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMinActive)},
    {SIP_MLNAME_CAST(sipName_GetMinXActive), meth_FbxLimits_GetMinXActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMinXActive)},
    {SIP_MLNAME_CAST(sipName_GetMinYActive), meth_FbxLimits_GetMinYActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMinYActive)},
    {SIP_MLNAME_CAST(sipName_GetMinZActive), meth_FbxLimits_GetMinZActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_GetMinZActive)},
    {SIP_MLNAME_CAST(sipName_SetActive), meth_FbxLimits_SetActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetActive)},
    {SIP_MLNAME_CAST(sipName_SetMax), meth_FbxLimits_SetMax, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMax)},
    {SIP_MLNAME_CAST(sipName_SetMaxActive), meth_FbxLimits_SetMaxActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMaxActive)},
    {SIP_MLNAME_CAST(sipName_SetMaxXActive), meth_FbxLimits_SetMaxXActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMaxXActive)},
    {SIP_MLNAME_CAST(sipName_SetMaxYActive), meth_FbxLimits_SetMaxYActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMaxYActive)},
    {SIP_MLNAME_CAST(sipName_SetMaxZActive), meth_FbxLimits_SetMaxZActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMaxZActive)},
    {SIP_MLNAME_CAST(sipName_SetMin), meth_FbxLimits_SetMin, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMin)},
    {SIP_MLNAME_CAST(sipName_SetMinActive), meth_FbxLimits_SetMinActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMinActive)},
    {SIP_MLNAME_CAST(sipName_SetMinXActive), meth_FbxLimits_SetMinXActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMinXActive)},
    {SIP_MLNAME_CAST(sipName_SetMinYActive), meth_FbxLimits_SetMinYActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMinYActive)},
    {SIP_MLNAME_CAST(sipName_SetMinZActive), meth_FbxLimits_SetMinZActive, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLimits_SetMinZActive)}
};

PyDoc_STRVAR(doc_FbxLimits, "\1FbxLimits()\n"
    "FbxLimits(FbxLimits)");


sipClassTypeDef sipTypeDef_fbx_FbxLimits = {
    {
        -1,
        0,
        0,
        SIP_TYPE_CLASS,
        sipNameNr_FbxLimits,
        {0},
        0
    },
    {
        sipNameNr_FbxLimits,
        {0, 0, 1},
        24, methods_FbxLimits,
        0, 0,
        0, 0,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    doc_FbxLimits,
    -1,
    -1,
    0,
    0,
    init_type_FbxLimits,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    dealloc_FbxLimits,
    assign_FbxLimits,
    array_FbxLimits,
    copy_FbxLimits,
    release_FbxLimits,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
